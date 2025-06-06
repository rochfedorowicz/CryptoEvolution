# tests/model/test_sklearn_model_adapter.py

# global imports
import logging
from ddt import ddt
from io import StringIO
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from unittest import TestCase
from unittest.mock import Mock, patch

# local imports
from source.model import SklearnModelAdapter

@ddt
class SklearnModelAdapterTestCase(TestCase):
    """
    Test case for SklearnModelAdapter class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")

        self.__mocked_estimator: BaseEstimator = Mock(spec = BaseEstimator)
        self.__sut: SklearnModelAdapter = SklearnModelAdapter(self.__mocked_estimator)

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

    @patch('joblib.load', new_callable = Mock)
    @patch('os.path.exists', new_callable = Mock)
    def test_sklearn_model_adapter_load_model__success(self, mocked_path_exists_function: Mock,
            mocked_joblib_load_function: Mock) -> None:
        """
        Tests the load_model method of the SklearnModelAdapter class.

        Verifies that the model is loaded correctly from the specified path.

        Asserts:
            The model is loaded correctly from the specified path.
            The model adapter's model attribute is set to the loaded estimator.
        """

        logging.info("Attempting to load model from path.")
        mocked_path_exists_function.return_value = True
        mocked_joblib_load_function.return_value = self.__mocked_estimator
        mocked_model_path = "mocked/path/to/model.pkl"

        logging.info("Invoking load_model method.")
        self.__sut.load_model(mocked_model_path)

        logging.info("Validating expected results.")
        mocked_path_exists_function.assert_called_once_with(mocked_model_path)
        mocked_joblib_load_function.assert_called_once_with(mocked_model_path)
        self.assertEqual(self.__sut._SklearnModelAdapter__model, self.__mocked_estimator)

    @patch('joblib.dump', new_callable = Mock)
    @patch('os.makedirs', new_callable = Mock)
    def test_sklearn_model_adapter_save_model__success(self, mocked_os_makedirs_function: Mock,
            mocked_joblib_dump_function: Mock) -> None:
        """
        Tests the save_model method of the SklearnModelAdapter class.

        Verifies that the model is saved correctly to the specified path.

        Asserts:
            The model is saved correctly to the specified path.
        """

        logging.info("Attempting to save model to path.")
        mocked_model_path = "mocked/path/to/model.pkl"

        logging.info("Invoking save_model method.")
        self.__sut.save_model(mocked_model_path)

        logging.info("Validating expected results.")
        mocked_os_makedirs_function.assert_called_once()
        mocked_joblib_dump_function.assert_called_once_with(self.__mocked_estimator, mocked_model_path)

    @patch('sys.stdout', new_callable = StringIO)
    def test_sklearn_model_adapter_print_summary(self, mocked_stdout: Mock) -> None:
        """
        Tests the print_summary method of the SklearnModelAdapter class.

        Verifies that the model's parameters are printed correctly.

        Asserts:
            Expected model parameters are printed to the standard output.
            The printed output matches the expected format.
        """

        logging.info("Attempting to print model summary.")
        self.__mocked_estimator.get_params.return_value = {
            'param1': 'value1',
            'param2': 'value2'
        }
        expected_summary = (
            "--------------------------------------------------------------------------------\n"
            "Model Summary: Mock\n"
            "--------------------------------------------------------------------------------\n"
            "Model Parameters:\n"
            "    param1: value1\n"
            "    param2: value2\n"
            "--------------------------------------------------------------------------------\n"
        )

        logging.info("Invoking print_summary method.")
        self.__sut.print_summary()

        logging.info("Validating expected results.")
        captured_output = mocked_stdout.getvalue().strip().split('\n')
        expected_lines = expected_summary.strip().split('\n')
        self.assertEqual(len(captured_output), len(expected_lines))
        for actual, expected in zip(captured_output, expected_lines):
            self.assertEqual(actual, expected)

    @patch.object(CalibratedClassifierCV, '__new__', new_callable = Mock)
    @patch('source.model.model_adapters.sklearn_model_adapter.learning_curve', new_callable = Mock)
    def test_sklearn_model_adapter_fit(self, mocked_learning_curve: Mock,
            mocked_calibrated_classifier_cv_constructor: Mock) -> None:
        """
        Tests the fit method of the SklearnModelAdapter class.

        Verifies that the model is trained correctly with the provided data.

        Asserts:
            The model's fit method is called with the correct input data.
            The model's score method is called with the correct validation data.
            The learning curve function is called with the correct parameters.
            The expected results are returned.
        """

        logging.info("Attempting to fit model with data.")
        mocked_score = 42
        mocked_train_sizes = [10, 20, 30]
        mocked_train_scores = [0.1, 0.2, 0.3]
        mocked_test_scores = [0.1, 0.2, 0.3]

        self.__mocked_estimator.fit = Mock()
        self.__mocked_estimator.score = Mock(return_value = mocked_score)
        mocked_learning_curve.return_value = (mocked_train_sizes, mocked_train_scores, mocked_test_scores)
        mocked_calibrated_classifier_cv_constructor.return_value = self.__mocked_estimator
        input_data = Mock()
        output_data = Mock()
        validation_input_data = Mock()
        validation_output_data = Mock()

        logging.info("Invoking fit method.")
        result = self.__sut.fit(input_data, output_data, (validation_input_data, validation_output_data))

        logging.info("Validating expected results.")
        self.__mocked_estimator.fit.assert_called_once_with(input_data, output_data)
        self.__mocked_estimator.score.assert_called_once_with(validation_input_data, validation_output_data)
        mocked_learning_curve.assert_called_once()
        self.assertEqual(result, {
            'learning_curve_data_train_sizes': mocked_train_sizes,
            'learning_curve_data_train_scores': mocked_train_scores,
            'learning_curve_data_valid_scores': mocked_test_scores,
            'validation_data_score': mocked_score
        })