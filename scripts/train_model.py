# scripts/train_model.py

# global imports
import argparse
import json
import logging
import os
import urllib.parse
import warnings
from datetime import datetime
from typing import Any

# local imports
from source.training import TrainingConfig, TrainingHandler
from source.utils import AWSHandler, DynamicFromStringConverter, GradientHandler

# suppress or filter out warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category = DeprecationWarning)
warnings.filterwarnings('ignore', category=  FutureWarning)
warnings.filterwarnings('ignore', category = UserWarning)

def __attempt_from_string_conversion(presumed_dict: Any) -> Any:
    if isinstance(presumed_dict, dict):
        class_name = presumed_dict.get('class_name', None)
        parameters = presumed_dict.get('parameters', None)
        class_type = presumed_dict.get('class_type', None)
        if (class_name is None or parameters is None) and class_type is None:
            logging.error("Failed to convert from string!")
            raise ValueError("Wrongly constructed configuration file! "
                             "Dictionaries must contain 'class_name' and 'parameters' keys "
                             "to be converted to an instance of a class or 'class_type' key to"
                             " be converted to a type.")

        if class_type is not None:
            return DynamicFromStringConverter().get_class_handle(class_type)

        for param_key, param_value in parameters.items():
            parameters[param_key] = __attempt_from_string_conversion(param_value)

        class_handle = DynamicFromStringConverter().get_class_handle(class_name)
        return class_handle(**parameters)

    elif isinstance(presumed_dict, list):
        return [__attempt_from_string_conversion(item) for item in presumed_dict]

    else:
        return presumed_dict

def __get_local_path(file_path: str) -> str:
        try:
            url_parsed = urllib.parse.urlparse(file_path)
            if url_parsed.netloc != '' and url_parsed.scheme != '':
                if url_parsed.scheme == 's3':
                    logging.info(f'Loading {file_path} from S3 bucket...')
                    aws_handler = AWSHandler()
                    file_name = '/'.join(file_path.split('/')[3:])
                    aws_handler.download_file_from_s3(os.getenv('BUCKET_NAME'), file_name)
                else:
                    logging.info(f'Loading {file_path} from public URL...')
                    response = urllib.request.urlopen(file_path)
                    response.raise_for_status()
                    local_file = open(file_path.split('/')[-1], 'wb')
                    local_file.write(response.read())
                local_path = os.getcwd() + '/' + file_path.split('/')[-1]
            else:
                logging.info(f'Loading {file_path} from local...')
                local_path = file_path
        except Exception as e:
            logging.error(f'Failed to localize {file_path}!')
            logging.error(e)
            raise e

        return local_path

def main(config_path: str, invoked_inside_gradient: bool = False) -> None:
    try:
        DynamicFromStringConverter().register_packages(['source',  'typing', 'imblearn', 'sklearn', 'tensorflow'])

        config_local_path = __get_local_path(config_path)
        config = json.load(open(config_local_path, 'r'))
        for key, value in config['training_config'].items():
            config['training_config'][key] = __attempt_from_string_conversion(value)

        data_set_name = config['data_set_name']
        config['training_config']['data_path'] = __get_local_path(data_set_name)

        callbacks = []
        callback_dicts_list = config.get('callbacks', None)
        if callback_dicts_list is not None:
            for callback_dict in callback_dicts_list:
                callbacks.append(__attempt_from_string_conversion(callback_dict))

        weights_load_path = None
        weights_file_name = config.get('weights_file_name', None)
        if weights_file_name is not None:
            weights_load_path = __get_local_path(weights_file_name)

        training_handler = TrainingHandler(TrainingConfig(**config['training_config']))
        training_handler.run_training(callbacks = callbacks, weights_load_path = weights_load_path)

        report_name = f"Report_{datetime.now().__format__('%Y-%m-%d_%H_%M_%S')}.pdf"
        report_path = os.getcwd() + '\\' + report_name
        training_handler.generate_report(report_path)
        aws_handler = AWSHandler()
        aws_handler.upload_file_to_s3(os.getenv('BUCKET_NAME'), report_path, report_name)

    except Exception as e:
        logging.error('Encounter problem during script execution!')
        logging.error(e)

    if invoked_inside_gradient:
        gradient_handler = GradientHandler()
        try:
            gradient_handler.delete_notebook(os.getenv('HOSTNAME'))
        except Exception as e:
            logging.error('Notebook was not deleted!')
            logging.error(e)

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "{asctime} | {levelname} | {message}",
                        style="{", datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description = 'Runs training described by configuration file.')
    parser.add_argument('--config_path', type = str, required = True,
                        help = 'Path to configuration file in *json format.')
    parser.add_argument('--gradient', action = 'store_true', default = False,
                        help = 'Indicates if it was run on a gradient notebook that should be closed at the end.')

    args = parser.parse_args()
    main(args.config_path, args.gradient)