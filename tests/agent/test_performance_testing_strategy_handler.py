# tests/agent/test_performance_testing_strategy_handler.py

# global imports
import logging
from ddt import ddt
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, PropertyMock

# local imports
from source.agent import PerformanceTestable, PerformanceTestingStrategyHandler
from source.environment import TradingEnvironment

@ddt
class PerformanceTestingStrategyHandlerTestCase(TestCase):
    """
    Test case for PerformanceTestingStrategyHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")

        self.__sut: PerformanceTestingStrategyHandler = PerformanceTestingStrategyHandler()

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

    def test_performance_testing_strategy_handler_evaluate(self) -> None:
        """
        Tests the evaluate method of the PerformanceTestingStrategyHandler.

        Verifies that the method correctly evaluates the performance of the agent
        over multiple iterations, collects the necessary data, and returns the expected keys
        and report data.

        Asserts:
            The keys returned match the expected plotting key.
            The report data contains the expected structure and values.
        """

        logging.info("Attempt to evaluate performance testing strategy handler.")
        mocked_testable_agent = Mock(spec = PerformanceTestable)
        mocked_environment = Mock(spec = TradingEnvironment)
        mocked_environment.state = SimpleNamespace()
        type(mocked_environment).current_iteration = PropertyMock(side_effect = [0, 2, 3, 4])
        mocked_environment.get_trading_data.return_value = SimpleNamespace(current_budget = 1000,
                                                                           currently_invested = 0)
        mocked_environment.step.side_effect = [
            [mocked_environment.state, 0, False, {'current_budget': 1000, 'currently_invested': 0}], # First step
            [mocked_environment.state, 0, False, {'current_budget': 800, 'currently_invested': 200}], # Second step
            [mocked_environment.state, 0, False, {'current_budget': 800, 'currently_invested': 220}], # Third step
            [mocked_environment.state, 0, False, {'current_budget': 800, 'currently_invested': 225}], # Fourth step
            [mocked_environment.state, 1, True, {'current_budget': 1230, 'currently_invested': 0}]  # Last step
        ]
        mocked_environment.get_data_for_iteration.return_value = [80000.0, 80000.0, 88000.0, 90000.0, 92000.0]
        expected_keys = [PerformanceTestingStrategyHandler.PLOTTING_KEY]
        expected_report_data = [{
            'assets_values': [1.0, 1.02, 1.025, 1.23],
            'reward_values': [0, 0, 0, 1],
            'currency_prices': [1.0, 1.0, 1.1, 1.125, 1.15],
            'infos': [{},
                      {'current_budget': 800, 'currently_invested': 220},
                      {'current_budget': 800, 'currently_invested': 225},
                      {'current_budget': 1230, 'currently_invested': 0}],
            'iterations': [0, 2, 3, 4],
            'solvency_coefficient': 57.5
        }]

        logging.info("Invoking evaluate.")
        keys, report_data = self.__sut.evaluate(mocked_testable_agent, mocked_environment)

        logging.info("Validating expected calls and results.")
        self.assertEqual(keys, expected_keys)
        self.assertEqual(report_data, expected_report_data)