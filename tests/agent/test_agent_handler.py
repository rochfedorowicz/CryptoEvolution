# tests/agent/test_agent_handler.py

# global imports
import logging
from ddt import ddt
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import ANY, Mock

# local imports
from source.agent import AgentBase, AgentHandler, LearningStrategyHandlerBase, TestingStrategyHandlerBase
from source.environment import TradingEnvironment
from source.model import BluePrintBase

@ddt
class AgentHandlerTestCase(TestCase):
    """
    Test case for AgentHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__mocked_environment: TradingEnvironment = Mock(spec = TradingEnvironment)
        self.__mocked_agent: AgentBase = Mock(spec = AgentBase)
        self.__mocked_learning_strategy_handler: LearningStrategyHandlerBase = Mock(spec = LearningStrategyHandlerBase)
        self.__mocked_learning_strategy_handler.create_agent.return_value = self.__mocked_agent
        self.__mocked_testing_strategy_handlers: list[TestingStrategyHandlerBase] = [Mock(spec = TestingStrategyHandlerBase)]

        self.__sut: AgentHandler = AgentHandler(Mock(spec = BluePrintBase), self.__mocked_environment,
                                                self.__mocked_learning_strategy_handler,
                                                self.__mocked_testing_strategy_handlers)

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


    def test_agent_handler_train_agent(self) -> None:
        """
        Tests AgentHandler's train_agent functionality.

        Verifies that the train_agent method correctly loads model, fits
        the underlying agent to the environment using learning strategy,
        and saves the model afterwards.

        Asserts:
            The agent's load_model method is called with the correct path.
            The environment's set_mode is called with TRAIN_MODE.
            The learning strategy handler's fit method is called with the correct parameters.
            The agent's save_model method is called with the correct path.
            The returned keys and report_data match the mocked values.
        """

        logging.info("Attempt to train agent.")
        mocked_keys = ['mocked_key1', 'mocked_key2']
        mocked_report_data = [{'mocked_metric1': 0.1}, {'mocked_metric2': 0.2}]
        self.__mocked_learning_strategy_handler.fit.return_value = (mocked_keys, mocked_report_data)
        nr_of_steps = 1000
        nr_of_episodes = 10
        model_load_path = "mock/path/to/load/model.ext"
        model_save_path = "mock/path/to/save/model.ext"
        callbacks = []

        logging.info("Invoking train_agent.")
        keys, report_data = self.__sut.train_agent(nr_of_steps, nr_of_episodes, callbacks = callbacks,
                                                   model_load_path = model_load_path,
                                                   model_save_path = model_save_path)

        logging.info("Validating expected calls and results.")
        self.__mocked_agent.load_model.assert_called_once_with(model_load_path)
        self.__mocked_environment.set_mode.assert_called_once_with(TradingEnvironment.TRAIN_MODE)
        self.__mocked_learning_strategy_handler.fit. \
            assert_called_once_with(self.__mocked_agent, self.__mocked_environment, nr_of_steps,
                                    nr_of_episodes, callbacks)
        self.__mocked_agent.save_model.assert_called_once_with(model_save_path)
        self.assertEqual(keys, mocked_keys)
        self.assertEqual(report_data, mocked_report_data)

    def test_agent_handler_test_agent__agent_not_fitted(self) -> None:
        """
        Tests AgentHandler's test_agent functionality when agent is not trained.

        Verifies that the test_agent method properly handles the case when an untrained
        agent attempts to run testing. It should detect that the agent is not trained
        and return empty results without calling the testing strategy.

        Asserts:
            The testing strategy handler's evaluate method is not called.
            Empty dictionaries are returned for both keys and report_data.
        """

        logging.info("Attempt to test agent without training.")
        self.__update_sut(_AgentHandler__trained = False)
        repeat = 2

        logging.info("Invoking test_agent.")
        keys, report_data = self.__sut.test_agent(repeat)

        logging.info("Validating expected calls and results.")
        for mocked_testing_strategy_handler in self.__mocked_testing_strategy_handlers:
            mocked_testing_strategy_handler.evaluate.assert_not_called()
        self.assertEqual(keys, {})
        self.assertEqual(report_data, {})

    def test_agent_handler_test_agent__agent_fitted_properly(self) -> None:
        """
        Tests AgentHandler's test_agent functionality when agent is properly trained.

        Verifies that the test_agent method correctly interacts with the environment
        when the agent has been trained. It should reset the environment, call the
        testing strategy evaluate method to get the results, and return the collected data.

        Asserts:
            The environment's set_mode is called with TEST_MODE.
            The environment's reset method is called exactly 'repeat' times.
            The testing strategy handler's evaluate method is called with correct parameters and exactly 'repeat' times.
            The returned keys dictionary contains the mocked keys for each repeat iteration.
            The returned report_data dictionary contains the mocked report data for each repeat iteration.
        """

        logging.info("Attempt to test agent without training.")
        self.__update_sut(_AgentHandler__trained = True)
        mocked_keys = ['mocked_key1', 'mocked_key2']
        mocked_report_data = [{'mocked_metric1': 0.11}, {'mocked_metric2': 0.21}]
        self.__mocked_testing_strategy_handlers[0].evaluate.return_value = (mocked_keys, mocked_report_data)
        self.__mocked_environment.get_environment_length.return_value = 10000
        self.__mocked_environment.get_trading_consts.return_value = SimpleNamespace(WINDOW_SIZE = 48)
        repeat = 2

        logging.info("Invoking test_agent.")
        keys, report_data = self.__sut.test_agent(repeat)

        logging.info("Validating expected calls and results.")
        self.__mocked_environment.set_mode.assert_called_once_with(TradingEnvironment.TEST_MODE)
        for mocked_testing_strategy_handler in self.__mocked_testing_strategy_handlers:
            mocked_testing_strategy_handler.evaluate.assert_called_with(self.__mocked_agent, self.__mocked_environment, ANY)
        self.assertEqual(self.__mocked_testing_strategy_handlers[0].evaluate.call_count, repeat)
        self.assertEqual(keys, {0: mocked_keys, 1: mocked_keys})
        self.assertEqual(report_data, {0: mocked_report_data, 1: mocked_report_data})
