# tests/agent/test_reinforcement_learning_strategy_handler.py

# global imports
import logging
import numpy as np
import rl
from ddt import ddt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

# local imports
from source.agent import LearningStrategyHandlerBase, ReinforcementLearningAgent, \
    ReinforcementLearningStrategyHandler
from source.environment import TradingEnvironment
from source.model import BluePrintBase, TFModelAdapter

@ddt
class ReinforcementLearningStrategyHandlerTestCase(TestCase):
    """
    Test case for ReinforcementLearningStrategyHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")

        self.__sut: ReinforcementLearningStrategyHandler = ReinforcementLearningStrategyHandler()

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

    @patch.object(rl.agents.DQNAgent, '__new__')
    def test_reinforcement_learning_strategy_handler_create_agent(self, mocked_dqn_agent_constructor: Mock) -> None:
        """
        Tests the creation of a reinforcement learning agent.

        Verifies that the agent is created with the correct parameters.

        Asserts:
            The agent is an instance of ReinforcementLearningAgent.
            The model adapter is instantiated with the correct parameters.
        """

        logging.info("Attempting to create agent.")
        mocked_dqn_agent_constructor.return_value = Mock(spec = rl.agents.DQNAgent)
        mocked_dqn_agent_constructor.return_value.model = Mock(spec = Model)
        mocked_dqn_agent_constructor.return_value.model.optimizer = Mock(spec = Optimizer)

        mocked_model_blue_print = Mock(spec = BluePrintBase)
        mocked_model_blue_print.report_parameters_needed_for_instantiation.return_value = \
            ['input_shape', 'output_length', 'spatial_data_shape']
        mocked_tf_model = Mock(spec = Model)
        mocked_tf_model.output_shape = (None, 3)
        mocked_tf_model_adapter = Mock(spec = TFModelAdapter)
        mocked_tf_model_adapter.get_model.return_value = mocked_tf_model
        mocked_model_blue_print.instantiate_model.return_value = mocked_tf_model_adapter

        mocked_trading_environment = Mock(spec = TradingEnvironment)
        expected_observation_space = SimpleNamespace(shape = (128,))
        expected_action_space = SimpleNamespace(n = 3)
        expected_spatial_data_shape = (1, 128)
        mocked_trading_environment.observation_space = expected_observation_space
        mocked_trading_environment.action_space = expected_action_space
        mocked_trading_environment.get_environment_spatial_data_dimension.return_value = expected_spatial_data_shape

        logging.info("Invoking create_agent method.")
        agent = self.__sut.create_agent(mocked_model_blue_print, mocked_trading_environment)

        logging.info("Validating expected results.")
        self.assertTrue(isinstance(agent, ReinforcementLearningAgent))
        mocked_model_blue_print.report_parameters_needed_for_instantiation.assert_called_once()
        mocked_model_blue_print.instantiate_model.assert_called_once_with(
            input_shape = expected_observation_space.shape,
            output_length = expected_action_space.n,
            spatial_data_shape = expected_spatial_data_shape
        )
        mocked_trading_environment.get_environment_spatial_data_dimension.assert_called_once()

    def test_reinforcement_learning_strategy_handler_fit(self) -> None:
        """
        Tests the fitting process of the reinforcement learning agent.

        Verifies that the agent is trained with the correct data and parameters.

        Asserts:
            The keys and data returned match the expected values.
        """

        logging.info("Attempting to train agent.")
        expected_nr_of_steps = 1000
        expected_nr_of_episodes = 10
        expected_callbacks = []
        mocked_fit_result = {
            'nb_steps': [100, 200, 300, 400, 500],
            'episode_reward': [1, 1, 0, -1, 1],
            'episode_loss': [0.7, 0.6, 1.2, 2.1, 0.7]
        }

        mocked_price_movement_train_data = np.array([80000.0, 80010.0, 80020.0, 80030.0, 80040.0])
        mocked_price_movement_test_data = np.array([80050.0, 80060.0, 80070.0, 80080.0, 80090.0])
        mocked_volatility_train_data = np.array([0.01, 0.02, 0.015, 0.025, 0.02])
        mocked_volatility_test_data = np.array([0.03, 0.025, 0.02, 0.015, 0.01])
        mocked_agent = Mock(spec = ReinforcementLearningAgent)
        mocked_agent.reinforcement_learning_fit.return_value = mocked_fit_result
        mocked_environment = Mock(spec = TradingEnvironment)
        mocked_environment.get_data_for_iteration.side_effect = [
            mocked_price_movement_test_data,
            mocked_volatility_test_data,
            mocked_price_movement_train_data,
            mocked_volatility_train_data
        ]

        expected_keys = [LearningStrategyHandlerBase.PLOTTING_KEY, ReinforcementLearningStrategyHandler.PLOTTING_KEY]
        expected_data = [
            {
                'test_part_price_movement': mocked_price_movement_test_data,
                'test_part_volatility': mocked_volatility_test_data,
                'train_part_price_movement': mocked_price_movement_train_data,
                'train_part_volatility': mocked_volatility_train_data
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
