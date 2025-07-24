# tests/model/test_tf_model_adapter.py

# global imports
import logging
import numpy as np
from ddt import ddt
from tensorflow.keras.models import Model
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

# local imports
from source.model import TFModelAdapter

@ddt
class TFModelAdapterTestCase(TestCase):
    """
    Test case for TFModelAdapter class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")

        self.__mocked_model: Model = Mock(spec = Model)
        self.__sut: TFModelAdapter = TFModelAdapter(self.__mocked_model)

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    def __update_sut(self, **kwargs) -> None:
        """
        Allows to update already created sut. It speeds up test
        cases' scenarios by enabling injections of certain values
        also into private sut members.
        """

        for name, value in kwargs.items():
            for attribute_name in self.__sut.__dict__:
                if name in attribute_name:
                    setattr(self.__sut, attribute_name, value)

    def test_tf_model_adapter_load_model__success(self) -> None:
        """
        Tests the load_model method of the TFModelAdapter class.

        Verifies that the model is loaded correctly from the specified path.

        Asserts:
            The model is loaded correctly from the specified path.
        """

        logging.info("Attempting to load model from path.")
        mocked_model_path = "mocked/path/to/model.h5"

        logging.info("Invoking load_model method.")
        self.__sut.load_model(mocked_model_path)

        logging.info("Validating expected results.")
        self.__mocked_model.load_weights.assert_called_once_with(mocked_model_path)

    def test_tf_model_adapter_save_model__success(self) -> None:
        """
        Tests the save_model method of the TFModelAdapter class.

        Verifies that the model is saved correctly to the specified path.

        Asserts:
            The model is saved correctly to the specified path.
        """

        logging.info("Attempting to save model to path.")
        mocked_model_path = "mocked/path/to/model.h5"

        logging.info("Invoking save_model method.")
        self.__sut.save_model(mocked_model_path)

        logging.info("Validating expected results.")
        self.__mocked_model.save_weights.assert_called_once_with(mocked_model_path)

    def test_tf_model_adapter_print_summary(self) -> None:
        """
        Tests the print_summary method of the TFModelAdapter class.

        Verifies that the model's parameters are printed correctly.

        Asserts:
            Summary was called on the model.
        """

        logging.info("Attempting to print model summary.")
        mocked_print_function = Mock()

        logging.info("Invoking print_summary method.")
        self.__sut.print_summary(mocked_print_function)

        logging.info("Validating expected results.")
        self.__mocked_model.summary.assert_called_once_with(print_fn = mocked_print_function)

    def test_tf_model_adapter_fit(self) -> None:
        """
        Tests the fit method of the TFModelAdapter class.

        Verifies that the model is trained correctly with the provided data.

        Asserts:
            The model's fit method is called with the correct input data.
            The model's score method is called with the correct validation data.
            The expected results are returned.
        """

        logging.info("Attempting to fit model with data.")
        self.__mocked_model.output_shape = (None, 2)
        expected_fit_return_value = {
            'loss': [0.1, 0.05],
            'accuracy': [0.95, 0.98]
        }
        self.__mocked_model.fit.return_value = SimpleNamespace(history = expected_fit_return_value)
        mocked_input_data = np.array([[1, 2], [3, 4]])
        mocked_output_data = np.array([[0], [1]])
        mocked_validation_data = (mocked_input_data, mocked_output_data)
        expected_input_data = np.array([[[1, 2]], [[3, 4]]])
        expected_output_data = np.array([[1, 0], [0, 1]]).astype(np.float32)
        expected_validation_data = (expected_input_data, expected_output_data)
        expected_epochs = 10
        expected_batch_size = 32
        expected_callbacks = []

        logging.info("Invoking fit method.")
        result = self.__sut.fit(mocked_input_data, mocked_output_data,
                                validation_data = mocked_validation_data, epochs = expected_epochs,
                                batch_size = expected_batch_size, callbacks = expected_callbacks)
        call_args = self.__mocked_model.fit.call_args

        logging.info("Validating expected results.")
        self.__mocked_model.fit.assert_called_once()
        self.assertEqual(call_args[0][0].tolist(), expected_input_data.tolist())
        self.assertEqual(call_args[0][1].tolist(), expected_output_data.tolist())
        self.assertEqual(call_args.kwargs['validation_data'][0].tolist(),
                         expected_validation_data[0].tolist())
        self.assertEqual(call_args.kwargs['validation_data'][1].tolist(),
                         expected_validation_data[1].tolist())
        self.assertEqual(call_args.kwargs['epochs'], expected_epochs)
        self.assertEqual(call_args.kwargs['batch_size'], expected_batch_size)
        self.assertEqual(call_args.kwargs['callbacks'], expected_callbacks)
        self.assertEqual(result, expected_fit_return_value)

    def test_tf_model_adapter_predict(self) -> None:
        """
        Tests the predict method of the TFModelAdapter class.

        Verifies that the model is used for making predictions.

        Asserts:
            The model's predict method is called with the correct input data.
            The expected results are returned.
        """

        logging.info("Attempting to predict with model.")
        expected_predict_return_value = {
            'class_0': 0.1,
            'class_1': 0.9
        }
        self.__mocked_model.predict.return_value = expected_predict_return_value
        mocked_input_data = np.array([[1, 2], [3, 4]])
        expected_input_data = np.array([[[1, 2]], [[3, 4]]])

        logging.info("Invoking predict method.")
        result = self.__sut.predict(mocked_input_data)
        call_args = self.__mocked_model.predict.call_args

        logging.info("Validating expected results.")
        self.__mocked_model.predict.assert_called_once()
        self.assertEqual(call_args[0][0].tolist(), expected_input_data.tolist())
        self.assertEqual(result, expected_predict_return_value)
