# tests/agent/test_classification_learning_agent.py

# global imports
import logging
from ddt import ddt
from unittest import TestCase
from unittest.mock import Mock

# local imports
from source.agent import ClassificationLearningAgent
from source.model import ModelAdapterBase

@ddt
class ClassificationLearningAgentTestCase(TestCase):
    """
    Test case for ClassificationLearningAgent class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__mocked_model_adapter: ModelAdapterBase = Mock(spec = ModelAdapterBase)
        self.__sut: ClassificationLearningAgent = ClassificationLearningAgent(self.__mocked_model_adapter)

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


    def test_classification_learning_agent_classification_fit(self) -> None:
        """
        Tests the classification_fit method of the ClassificationLearningAgent.

        Verifies that the method correctly calls the fit method of the model adapter
        with the expected parameters and handles the training process correctly.

        Asserts:
            The fit method of the model adapter is called with the correct parameters.
            The training history is correctly processed and returned.
        """

        logging.info("Attempting to fit the model.")
        expected_history = {'loss': 0.1, 'accuracy': 0.95}
        self.__mocked_model_adapter.fit.return_value = expected_history
        self.__mocked_model_adapter.report_parameters_needed_for_fitting.return_value = \
            ['input_data', 'output_data', 'validation_data']
        mocked_input_data = Mock()
        mocked_output_data = Mock()
        mocked_validation_data = Mock()

        logging.info("Invoking classification_fit.")
        result = self.__sut.classification_fit(
            mocked_input_data,
            mocked_output_data,
            validation_data = mocked_validation_data,
            batch_size = 32,
            epochs = 10,
            callbacks = []
        )

        logging.info("Validating expected calls and results.")
        self.__mocked_model_adapter.fit.assert_called_once_with(
            input_data = mocked_input_data,
            output_data = mocked_output_data,
            validation_data = mocked_validation_data
        )
        self.assertEqual(result, expected_history)