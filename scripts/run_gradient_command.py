# scripts/run_gradient_command.py

# global imports
import argparse
import logging
from datetime import datetime
from typing import Optional

# local imports
from source.utils import GradientHandler

def __str_to_dict(environment_dict_str: str) -> dict:
    environment_dict = dict()

    if not environment_dict_str:
        return environment_dict

    for environmental_variable in environment_dict_str.replace(' ', '').split(','):
        key, value = environmental_variable.split(':')
        environment_dict[key] = value

    return environment_dict

def main(command: str, url: Optional[str] = None, notebook_name: Optional[str] = None, machines: Optional[str] = None,
         timeout: Optional[int] = None, environment: Optional[str] = None) -> None:
    if timeout is None:
        logging.info('No timeout set, using 6 hours as default.')
        timeout = 6

    if machines is None:
        logging.info('No machines specified, using Free-P5000 as default.')
        machines = ['Free-P5000']
    else:
        machines = machines.split(',')

    if notebook_name is None:
        notebook_name = f"Training notebook-{datetime.now().__format__('%Y-%m-%d_%H:%M:%S')}"
        logging.info(f'No name specified, using {notebook_name} as default.')

    try:
        gradient_handler = GradientHandler()
        notebook_id = gradient_handler.create_notebook(command, url, notebook_name, machines,
                                                       timeout, __str_to_dict(environment))
        if notebook_id is not None:
            logging.info(f'Instantiated training run successfully on notebook {notebook_id}.')

    except Exception as e:
        logging.error('Encounter problem during script execution!')
        logging.error(e)

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "{asctime} | {levelname} | {message}",
                        style="{", datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description = 'Instantiate training on Paperspace Gradient machine.')
    parser.add_argument('--command', type = str, required = True, help = 'Command invoked at machine.')
    parser.add_argument('--url', type = str, required = False,
                        help = 'Url to github repository that should be copied into machine.')
    parser.add_argument('--name', type = str, required = False, help = 'Name of the notebook.')
    parser.add_argument('--machines', type = str, required = False,
                        help = '''List of machines, that looks like: machine_type_1,machine_type_2,...,machine_type_N.
                        Possible machine types are: e.g. Free-P5000, Free-A4000, Free-RTX4000, Free-RTX5000.''')
    parser.add_argument('--timeout', type = int, required = False,
                        help = 'Timeout that machine will be terminated after. Maximal value is 6.')
    parser.add_argument('--env', type = str, required = False,
                        help = 'String denoted dictionary of environmental variables, eg. key:value,key_2:value_2.')

    args = parser.parse_args()
    main(args.command, args.url, args.name, args.machines, args.timeout, args.env)