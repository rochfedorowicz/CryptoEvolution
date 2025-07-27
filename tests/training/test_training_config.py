# tests/training/test_training_config.py

# global imports
import logging
import pandas as pd
from ddt import data, ddt, unpack
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, patch

# local imports
from source.agent import AgentHandler, LearningStrategyHandlerBase, TestingStrategyHandlerBase
from source.environment import TradingEnvironment
from source.model import BluePrintBase
from source.training import TrainingConfig

# file constants
INITIAL_BUDGET = 1000.0
MAX_AMOUNT_OF_TRADES = 5
WINDOW_SIZE = 48

@ddt
class TrainingConfigTestCase(TestCase):
    """
    Test case for TrainingConfig class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        nr_of_steps = 2000
        nr_of_episodes = 100
        self.__mocked_model_blue_print = Mock(spec = BluePrintBase)
        self.__mocked_learning_strategy_handler = Mock(spec = LearningStrategyHandlerBase)

        self.__sut: TrainingConfig = TrainingConfig(nr_of_steps = nr_of_steps,
                                                    nr_of_episodes = nr_of_episodes,
                                                    model_blue_print = self.__mocked_model_blue_print,
                                                    data = Mock(spec = pd.DataFrame),
                                                    initial_budget = INITIAL_BUDGET,
                                                    max_amount_of_trades = MAX_AMOUNT_OF_TRADES,
                                                    window_size = WINDOW_SIZE,
                                                    learning_strategy_handler = self.__mocked_learning_strategy_handler,
                                                    testing_strategy_handler = Mock(spec = TestingStrategyHandlerBase))

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

    @data(
        ({
            'nr_of_steps': 3000,
            'nr_of_episodes': 200,
            'initial_budget': 1500.0,
            'max_amount_of_trades': 10,
            'window_size': 72
        },
            "Training config:\n"
            "\tnr_of_steps: 3000\n"
            "\tnr_of_episodes: 200\n"
            "\trepeat_test: 10\n"
            "\ttest_ratio: 0.2\n"
            "\tinitial_budget: 1500.0\n"
            "\tmax_amount_of_trades: 10\n"
            "\twindow_size: 72\n"
            "\tsell_stop_loss: 0.8\n"
            "\tsell_take_profit: 1.2\n"
            "\tbuy_stop_loss: 0.8\n"
            "\tbuy_take_profit: 1.2\n"
            "\tpenalty_starts: 0\n"
            "\tpenalty_stops: 10\n"
            "\tstatic_reward_adjustment: 1\n"
            "\tvalidator: PriceRewardValidator\n"
            "\t\t{'_PriceRewardValidator__coefficient': 1.0, '_PriceRewardValidator__normalizable': False}\n"
            "\tlabel_annotator: SimpleLabelAnnotator\n"
            "\t\t{'_output_classes': namespace(UP_TREND=0, DOWN_TREND=1, NO_TREND=2), '_requested_columns': None, '_SimpleLabelAnnotator__threshold': 0.01}\n"
            "\tlabeled_data_balancer: None\n"
            "\tmodel_blue_print: BluePrintBase\n"
            "\t\t{}\n"
            "\tlearning_strategy_handler: LearningStrategyHandlerBase\n"
            "\t\t{}\n"
            "\ttesting_strategy_handler: TestingStrategyHandlerBase\n"
            "\t\t{}\n")
    )
    @unpack
    def test_training_config___str__(self, params_to_update: dict[str, Any], expected_serialization_str: str) -> None:
        """
        Tests TrainingConfig's __str__ functionality.

        Verifies that the string representation of the TrainingConfig object is correct
        after updating various parameters. The test sanitizes memory location identifiers
        to ensure consistent comparison.

        Parameters:
            params_to_update: Dictionary with parameters to update in the training config.
            expected_serialization_str: Expected string representation after updates.

        Asserts:
            The sanitized string representation matches the expected string.
        """

        logging.info("Attempting to serialize TrainingConfig.")
        self.__update_sut(**params_to_update)

        with patch('builtins.vars', side_effect = lambda obj: {} if isinstance(obj, Mock) else obj.__dict__):
            logging.info("Serializing TrainingConfig with updated parameters.")
            result = str(self.__sut)

        logging.info("Validating expected result.")
        self.assertEqual(result, expected_serialization_str)

    @patch.object(TradingEnvironment, '__new__')
    def test_training_config_instantiate_agent_handler(self, mocked_environment_constructor: Mock) -> None:
        """
        Tests TrainingConfig's instantiate_agent_handler functionality.

        Verifies that the instantiate_agent_handler method correctly creates an AgentHandler.

        Asserts:
            The created AgentHandler is associated with the mocked environment.
        """

        logging.info("Attempting to instantiate agent from TrainingConfig.")
        mocked_environment_constructor.return_value = Mock(spec = TradingEnvironment)

        logging.info("Invoking instantiate_agent_handler.")
        agent_handler = self.__sut.instantiate_agent_handler()

        logging.info("Validating expected calls and results.")
        self.__mocked_learning_strategy_handler. \
            create_agent.assert_called_once_with(self.__mocked_model_blue_print,
                                                 mocked_environment_constructor.return_value)
        self.assertTrue(isinstance(agent_handler, AgentHandler))
