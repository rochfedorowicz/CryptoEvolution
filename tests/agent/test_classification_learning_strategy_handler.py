# tests/agent/test_classification_learning_strategy_handler.py

# global imports
import logging
import numpy as np
from ddt import ddt
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

# local imports
from source.agent import ClassificationLearningAgent, ClassificationLearningStrategyHandler, \
    LearningStrategyHandlerBase
from source.environment import TradingEnvironment
from source.model import BluePrintBase

@ddt
class ClassificationLearningStrategyHandlerTestCase(TestCase):
    """
    Test case for ClassificationLearningStrategyHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")

        self.__sut: ClassificationLearningStrategyHandler = ClassificationLearningStrategyHandler()

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

    def test_classification_learning_strategy_handler_create_agent(self) -> None:
        """
        Tests the create_agent method of the ClassificationLearningStrategyHandler.

        Verifies that the method correctly creates an instance of ClassificationLearningAgent
        with the expected parameters and returns it.

        Asserts:
            The created agent is an instance of ClassificationLearningAgent.
            The model adapter is instantiated with the correct parameters.
            The environment's spatial data dimension and trading constants are correctly retrieved.
        """

        logging.info("Attempting to create agent.")
        mocked_model_blue_print = Mock(spec = BluePrintBase)
        mocked_model_blue_print.report_parameters_needed_for_instantiation.return_value = \
            ['input_shape', 'output_length', 'spatial_data_shape']
        mocked_trading_environment = Mock(spec = TradingEnvironment)
        expected_window_size = 12
        expected_spatial_data_shape = (expected_window_size, 128)
        mocked_output_classes = ['class1', 'class2', 'class3']
        mocked_trading_environment.get_environment_spatial_data_dimension.return_value = expected_spatial_data_shape
        mocked_trading_environment.get_trading_consts.return_value = SimpleNamespace(
            WINDOW_SIZE = expected_window_size,
            OUTPUT_CLASSES = mocked_output_classes
        )

        logging.info("Invoking create_agent method.")
        agent = self.__sut.create_agent(mocked_model_blue_print, mocked_trading_environment)

        logging.info("Validating expected results.")
        self.assertTrue(isinstance(agent, ClassificationLearningAgent))
        mocked_model_blue_print.report_parameters_needed_for_instantiation.assert_called_once()
        mocked_model_blue_print.instantiate_model.assert_called_once_with(
            input_shape = (expected_spatial_data_shape[1] * expected_window_size, ),
            output_length = len(mocked_output_classes),
            spatial_data_shape = expected_spatial_data_shape
        )
        self.assertEqual(mocked_trading_environment.get_environment_spatial_data_dimension.call_count, 2)
        self.assertEqual(mocked_trading_environment.get_trading_consts.call_count, 2)

    def test_classification_learning_strategy_handler_fit(self) -> None:
        """
        Tests the fit method of the ClassificationLearningStrategyHandler.

        Verifies that the method correctly processes the training data, calls the agent's
        classification_fit method, and returns the expected keys and data.

        Asserts:
            The returned keys and data match the expected values.
        """

        logging.info("Attempting to train agent.")
        expected_nr_of_steps = 1000
        expected_nr_of_episodes = 10
        expected_callbacks = []
        mocked_adapter_tag = "mocked"
        mocked_fit_result = {
            'loss': 0.1,
            'accuracy': 0.95,
            'val_loss': 0.15,
            'val_accuracy': 0.90
        }
        mocked_price_movement_train_data = np.array([80000.0, 80010.0, 80020.0, 80030.0, 80040.0])
        mocked_price_movement_test_data = np.array([80050.0, 80060.0, 80070.0, 80080.0, 80090.0])
        mocked_volatility_train_data = np.array([0.01, 0.02, 0.015, 0.025, 0.02])
        mocked_volatility_test_data = np.array([0.03, 0.025, 0.02, 0.015, 0.01])
        mocked_label_train_data = np.array([0, 1, 0, 1, 0])
        mocked_label_test_data = np.array([1, 0, 1, 0, 1])
        mocked_agent = Mock(spec = ClassificationLearningAgent)
        mocked_agent.get_model_adapter_tag.return_value = mocked_adapter_tag
        mocked_agent.classification_fit.return_value = mocked_fit_result
        mocked_environment = Mock(spec = TradingEnvironment)
        mocked_environment.get_labeled_data.side_effect = [
            [
                np.array([[0.1], [0.4], [0.5], [0.6], [0.7]]),  # input_data
                mocked_label_train_data,                        # output_data
                np.array([[0.1], [0.4], [0.5], [0.6], [0.7]]),  # input_data_test
                mocked_label_train_data                         # output_data_test
            ],
            [
                np.array([[0.2], [0.3], [0.4], [0.5], [0.6]]),  # input_data
                mocked_label_test_data,                         # output_data
                np.array([[0.2], [0.3], [0.4], [0.5], [0.6]]),  # input_data_test
                mocked_label_test_data                          # output_data_test
            ],
            [
                np.array([[0.1], [0.4], [0.5], [0.6], [0.7]]),  # input_data
                mocked_label_train_data,                        # output_data
                np.array([[0.1], [0.4], [0.5], [0.6], [0.7]]),  # input_data_test
                mocked_label_train_data                         # output_data_test
            ]
        ]
        mocked_environment.get_data_for_iteration.side_effect = [
            mocked_price_movement_test_data,
            mocked_volatility_test_data,
            mocked_price_movement_train_data,
            mocked_volatility_train_data,
            mocked_price_movement_test_data,
            mocked_price_movement_train_data
        ]

        expected_keys = [
            LearningStrategyHandlerBase.PLOTTING_KEY,
            ClassificationLearningStrategyHandler.PLOTTING_KEYS[0],
            f'{ClassificationLearningStrategyHandler.PLOTTING_KEYS[1]}_{mocked_adapter_tag}'
        ]
        expected_data = [
            {
                'test_part_price_movement': mocked_price_movement_test_data,
                'test_part_volatility': mocked_volatility_test_data,
                'train_part_price_movement': mocked_price_movement_train_data,
                'train_part_volatility': mocked_volatility_train_data
            },
            {
                'test_part_price_movement': mocked_price_movement_test_data,
                'test_part_labels': mocked_label_test_data,
                'train_part_price_movement': mocked_price_movement_train_data,
                'train_part_labels': mocked_label_train_data
            },
            mocked_fit_result
        ]

        logging.info("Invoking fit method.")
        keys, data = self.__sut.fit(mocked_agent, mocked_environment, expected_nr_of_steps,
                                    expected_nr_of_episodes, expected_callbacks)

        logging.info("Validating expected results.")
        self.assertEqual(keys, expected_keys)
        for data_item, expected_data_item in zip(data, expected_data):
            for expected_key, expected_value in expected_data_item.items():
                self.assertIn(expected_key, data_item)
                if expected_key in data_item.keys():
                    if isinstance(expected_value, np.ndarray):
                        self.assertEqual(expected_value.tolist(), data_item[expected_key].tolist())
                    else:
                        self.assertEqual(expected_value, data_item[expected_key])
