# tests/environment/test_trading_environment.py

# global imports
import logging
import numpy as np
import pandas as pd
from unittest import TestCase
from unittest.mock import Mock, patch
from types import SimpleNamespace

# local imports
from source.environment import Broker, LabelAnnotatorBase, LabeledDataBalancer, Order, \
    RewardValidatorBase, TradingEnvironment

class TradingEnvironmentTestCase(TestCase):
    """
    Test case TradingEnvironment class. Stores all the test cases
    and allows for convenient test case execution.
    """

    # local constants
    __MOCKED_CSV_DATA = pd.DataFrame(data = {
        'low': [20000.0, 20500.0, 20100.0, 20100.0, 20000.0],
        'high': [20900.0, 20900.0, 21000.0, 20900.0, 21700.0],
        'open': [20050.0, 20600.0, 20400.0, 20800.0, 20200.0],
        'close': [20600.0, 20400.0, 20800.0, 20200.0, 20900.0],
        'volume': [1000.0, 1200.0, 1100.0, 1300.0, 900.0]
    }, index = pd.DatetimeIndex(['2020-03-01', '2020-03-02', '2020-03-03',
                                 '2020-03-04', '2020-03-05'], name = 'time'))

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.

        Parameters:
            mock_pd_read_csv (Mock): Mock for pandas read csv function.
                Enables easier data loading for tests.
        """

        logging.info("Setting up test environment.")

        data = self.__MOCKED_CSV_DATA
        initial_budget = 1000.0
        max_amount_of_trades = 5
        window_size = 2
        sell_stop_loss = 0.95
        sell_take_profit = 1.05
        buy_stop_loss = sell_stop_loss
        buy_take_profit = sell_take_profit
        test_ratio = 0.0
        penalty_starts = 2
        penalty_stops = 4
        static_reward_adjustment = 1

        self.__mocked_reward_validator: RewardValidatorBase = Mock(spec = RewardValidatorBase)
        self.__mocked_reward_validator.validate_orders = lambda orders: \
            np.sum([order.current_value - order.initial_value for order in orders])
        self.__mocked_label_annotator: LabelAnnotatorBase = Mock(spec = LabelAnnotatorBase)
        self.__mocked_data_balancer: LabeledDataBalancer = Mock(spec = LabeledDataBalancer)
        self.__sut: TradingEnvironment = TradingEnvironment(data, initial_budget, max_amount_of_trades,
                                                            window_size, self.__mocked_reward_validator,
                                                            self.__mocked_label_annotator, sell_stop_loss,
                                                            sell_take_profit, buy_stop_loss, buy_take_profit,
                                                            test_ratio, penalty_starts, penalty_stops,
                                                            static_reward_adjustment, self.__mocked_data_balancer,
                                                            should_prefetch = False)

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    def __update_sut(self, **kwargs) -> None:
        """
        Allows to update already created sut. It speeds up test
        cases' scenarios by enabling injecting certain values
        also into private sut members.
        """

        for name, value in kwargs.items():
            for attribute_name in self.__sut.__dict__:
                attribute_value = getattr(self.__sut, attribute_name)

                if isinstance(attribute_value, SimpleNamespace):
                    if name in vars(attribute_value):
                        setattr(attribute_value, name, value)

                elif isinstance(attribute_value, Broker):
                    for broker_attribute_name in attribute_value.__dict__:
                        if name in broker_attribute_name:
                            setattr(attribute_value, broker_attribute_name, value)

                elif name in attribute_name:
                    setattr(self.__sut, attribute_name, value)

    def test_traiding_environment_create(self) -> None:
        """
        Tests TradingEnvironment's __init__ function.

        Verifies that trading environment is created correctly
        and all the needed functions as well as observation state
        are calculated properly.

        Asserts:
            Expected state equals current observation state. All the functions
            work correctly within typical input ranges.
        """

        logging.info("Starting creation test case.")
        traiding_consts = self.__sut.get_trading_consts()

        expected_state = [-1,   # normalized low[0] -> lower value from 20000.0 and 20500.0
                           0,   # normalized high[0] -> equal value from 20900.0 and 20900.0
                          -1,   # normalized open[0] -> lower value from 20050.0 and 20600.0
                           1,   # normalized close[0] -> higher value from 20600.0 and 20400.0
                          -1,   # normalized volume[0] -> lower value from 1000.0 and 1200.0
                           1,   # normalized low[1] -> higher value from 20000.0 and 20500.0
                           0,   # normalized high[1] -> equal value from 20900.0 and 20900.0
                           1,   # normalized open[1] -> higher value from 20050.0 and 20600.0
                          -1,   # normalized close[1] -> lower value from 20600.0 and 20400.0
                           1,   # normalized volume[1] -> higher value from 1000.0 and 1200.0
                           0,   # current_profitability_coeff -> current budget equal initial
                           0,   # current_no_trades_penalty_coeff -> no penalty
                           0,   # current_long_trades_occupancy_coeff -> no long trades placed
                           0]   # current_short_trades_occupancy_coeff -> no short trades placed

        logging.info("Checking created observation state.")
        assert [round(observation, 0) for observation in self.__sut.state] == expected_state

        logging.info("Checking created profitability function.")
        assert traiding_consts.PROFITABILITY_FUNCTION(2) > 0
        assert traiding_consts.PROFITABILITY_FUNCTION(1) == 0
        assert traiding_consts.PROFITABILITY_FUNCTION(0) < 0

        logging.info("Checking created penalty function.")
        for day in range(0, traiding_consts.PENALTY_STOPS + 1):
            if day < traiding_consts.PENALTY_STARTS:
                assert round(traiding_consts.PENALTY_FUNCTION(day), 2) == 0
            elif day >= traiding_consts.PENALTY_STARTS and \
                 day < traiding_consts.PENALTY_STOPS:
                assert round(traiding_consts.PENALTY_FUNCTION(day), 3) > 0
                assert round(traiding_consts.PENALTY_FUNCTION(day), 3) < 1
            else:
                assert round(traiding_consts.PENALTY_FUNCTION(day), 2) == 1

    def test_traiding_environment_step__action_successful_buy_sell(self) -> None:
        """
        Tests TradingEnvironment's step function - sucessful non-waiting actions.

        Verifies step execution with non-waiting actions when they are expected.
        In this case, before executing step function there is no ongoing trades,
        so each correctly performed non-waiting action should be rewarded with static
        reward adjusment.

        Asserts:
            Each of two actions is corrctly performed. The environment is correctly
            updated after each action and correct reward is returned.
        """

        logging.info("Starting step test case.")
        traiding_consts = self.__sut.get_trading_consts()

        expected_money_to_be_spent_on_trade = 200
        expected_budget_after_buy = 800
        expected_budget_after_sell = 600
        expected_invested_after_buy = 200
        expected_invested_after_sell = 400
        expected_no_trades_for = 0
        expected_nr_of_long_trades_after_buy = 1
        expected_nr_of_short_trades_after_buy = 0
        expected_nr_of_long_trades_after_sell = 1
        expected_nr_of_short_trades_after_sell = 1

        logging.info("Performing buy action.")
        buy_action = 0
        _, reward, _, step_info = self.__sut.step(buy_action)

        logging.info("Checking step info for successful buy.")
        assert step_info['action'] == buy_action
        assert step_info['money_to_trade'] == expected_money_to_be_spent_on_trade
        assert step_info['current_budget'] == expected_budget_after_buy
        assert step_info['currently_invested'] == expected_invested_after_buy
        assert step_info['no_trades_placed_for'] == expected_no_trades_for
        assert step_info['currently_placed_long_trades'] == expected_nr_of_long_trades_after_buy
        assert step_info['currently_placed_short_trades'] == expected_nr_of_short_trades_after_buy
        assert round(reward, 0) == traiding_consts.STATIC_REWARD_ADJUSTMENT

        logging.info("Performing sell action.")
        sell_action = 2
        _, reward, _, step_info = self.__sut.step(sell_action)

        logging.info("Checking step info for successful sell.")
        assert step_info['action'] == sell_action
        assert step_info['money_to_trade'] == expected_money_to_be_spent_on_trade
        assert step_info['current_budget'] == expected_budget_after_sell
        assert step_info['currently_invested'] == expected_invested_after_sell
        assert step_info['no_trades_placed_for'] == expected_no_trades_for
        assert step_info['currently_placed_long_trades'] == expected_nr_of_long_trades_after_sell
        assert step_info['currently_placed_short_trades'] == expected_nr_of_short_trades_after_sell
        assert round(reward, 0) == traiding_consts.STATIC_REWARD_ADJUSTMENT

    def test_traiding_environment_step__action_successful_wait(self) -> None:
        """
        Tests TradingEnvironment's step function - sucessful waiting action.

        Verifies step execution with waiting action when it is expected.
        In this case, before executing step function number of ongoing trades
        equals max possible number of trades to be placed, so wait action is
        the only one expected and should be rewarded with static reward adjusment.

        Asserts:
            Action is corrctly performed. The environment is correctly
            updated after action and correct reward is returned.
        """

        logging.info("Starting step test case.")
        traiding_consts = self.__sut.get_trading_consts()
        self.__update_sut(currently_placed_long_trades = traiding_consts.MAX_AMOUNT_OF_TRADES)

        expected_money_to_be_spent_on_trade = 0
        expected_budget_after_wait = 1000
        expected_invested_after_wait = 0
        expected_no_trades_for = 1
        expected_nr_of_long_trades_after_wait = traiding_consts.MAX_AMOUNT_OF_TRADES
        expected_nr_of_short_trades_after_wait = 0

        logging.info("Performing wait action.")
        wait_action = 1
        _, reward, _, step_info = self.__sut.step(wait_action)

        logging.info("Checking step info for successful wait.")
        assert step_info['action'] == wait_action
        assert step_info['money_to_trade'] == expected_money_to_be_spent_on_trade
        assert step_info['current_budget'] == expected_budget_after_wait
        assert step_info['currently_invested'] == expected_invested_after_wait
        assert step_info['no_trades_placed_for'] == expected_no_trades_for
        assert step_info['currently_placed_long_trades'] == expected_nr_of_long_trades_after_wait
        assert step_info['currently_placed_short_trades'] == expected_nr_of_short_trades_after_wait
        assert round(reward, 0) == traiding_consts.STATIC_REWARD_ADJUSTMENT

    def test_traiding_environment_step__action_failure_buy_sell(self) -> None:
        """
        Tests TradingEnvironment's step function - failure non-waiting actions.

        Verifies step execution with non-waiting actions when they are not expected.
        In this case, before executing step function number of ongoing trades
        equals max possible number of trades to be placed, so each performed non-waiting
        action should be rewarded with negative static reward adjusment. Assumes IMPLICIT_ORDER_CLOSING.

        Asserts:
            Each of two actions is corrctly performed. The environment is correctly
            updated after each action and correct reward is returned.
        """

        logging.info("Starting step test case.")
        traiding_consts = self.__sut.get_trading_consts()
        self.__update_sut(currently_placed_long_trades = traiding_consts.MAX_AMOUNT_OF_TRADES)

        expected_money_to_be_spent_on_trade = 0
        expected_budget_after_buy = 1000
        expected_budget_after_sell = 1000
        expected_invested_after_buy = 0
        expected_invested_after_sell = 0
        expected_no_trades_for_after_buy = 1
        expected_no_trades_for_after_sell = 2
        expected_nr_of_long_trades = traiding_consts.MAX_AMOUNT_OF_TRADES
        expected_nr_of_short_trades = 0

        logging.info("Performing buy action.")
        buy_action = 0
        _, reward, _, step_info = self.__sut.step(buy_action)

        logging.info("Checking step info for failure buy.")
        assert step_info['action'] == buy_action
        assert step_info['money_to_trade'] == expected_money_to_be_spent_on_trade
        assert step_info['current_budget'] == expected_budget_after_buy
        assert step_info['currently_invested'] == expected_invested_after_buy
        assert step_info['no_trades_placed_for'] == expected_no_trades_for_after_buy
        assert step_info['currently_placed_long_trades'] == expected_nr_of_long_trades
        assert step_info['currently_placed_short_trades'] == expected_nr_of_short_trades
        assert round(reward, 0) == -traiding_consts.STATIC_REWARD_ADJUSTMENT

        logging.info("Performing sell action.")
        sell_action = 2
        _, reward, _, step_info = self.__sut.step(sell_action)

        logging.info("Checking step info for failure sell.")
        assert step_info['action'] == sell_action
        assert step_info['money_to_trade'] == expected_money_to_be_spent_on_trade
        assert step_info['current_budget'] == expected_budget_after_sell
        assert step_info['currently_invested'] == expected_invested_after_sell
        assert step_info['no_trades_placed_for'] == expected_no_trades_for_after_sell
        assert step_info['currently_placed_short_trades'] == expected_nr_of_short_trades
        assert step_info['currently_placed_long_trades'] == expected_nr_of_long_trades
        assert round(reward, 0) == -traiding_consts.STATIC_REWARD_ADJUSTMENT

    def test_traiding_environment_step__action_failure_wait(self) -> None:
        """
        Tests TradingEnvironment's step function - failure waiting action.

        Verifies step execution with waiting action when it is not expected.
        In this case, before executing step function there is one ongoing trade
        and there was no trades placed for number of days bigger than penalty stop
        constant, so performed waiting action should be rewarded with negative
        static reward adjusment.

        Asserts:
            Action is corrctly performed. The environment is correctly
            updated after action and correct reward is returned.
        """

        logging.info("Starting step test case.")
        traiding_consts = self.__sut.get_trading_consts()
        orders = [Order(200, True, 0.99, 1.01)]
        self.__update_sut(current_orders = orders,
                          currently_placed_long_trades = 1,
                          currently_placed_short_trades = 0,
                          current_budget = 800,
                          currently_invested = 200,
                          no_trades_placed_for = traiding_consts.PENALTY_STOPS + 1)

        expected_money_to_be_spent_on_trade = 200
        expected_budget_after_wait = 1000
        expected_invested_after_wait = 0
        expected_no_trades_for_after_wait = traiding_consts.PENALTY_STOPS + 2
        expected_nr_of_long_trades = 0
        expected_nr_of_short_trades = 0

        logging.info("Performing wait action.")
        wait_action = 1
        _, reward, _, step_info = self.__sut.step(wait_action)

        logging.info("Checking step info for failure wait.")
        assert step_info['action'] == wait_action
        assert round(step_info['money_to_trade'], -1) == expected_money_to_be_spent_on_trade
        assert round(step_info['current_budget'], -1) == expected_budget_after_wait
        assert step_info['currently_invested'] == expected_invested_after_wait
        assert step_info['no_trades_placed_for'] == expected_no_trades_for_after_wait
        assert step_info['currently_placed_long_trades'] == expected_nr_of_long_trades
        assert step_info['currently_placed_short_trades'] == expected_nr_of_short_trades
        assert round(reward, 0) == -traiding_consts.STATIC_REWARD_ADJUSTMENT

    def test_traiding_environment_step__scenario_winning_trades(self) -> None:
        """
        Tests TradingEnvironment's step function - scenario winning trades.

        Verifies that reward is correctly calculated for winning trades, when
        no penalty is applied.

        Asserts:
            Coefficient is correctly calculated. The environment is correctly
            updated after action and correct reward is returned.
        """

        logging.info("Starting step scenario test case.")
        orders = [Order(200, True, 0.95, 1.05), Order(200, True, 0.95, 1.05)]
        self.__update_sut(leverage = 10,
                          current_orders = orders,
                          currently_placed_trades = 2,
                          current_budget = 600,
                          currently_invested = 400)

        expected_budget_after_wait_at_least = 1000
        expected_invested_after_wait = 0
        expected_coeff = (20800.0 - 20400.0) / 20400.0
        expected_orders_increase = [order.initial_value * expected_coeff for order in orders]
        expected_reward = np.sum(expected_orders_increase) * self.__sut.get_broker().get_leverage()

        logging.info("Performing wait action.")
        wait_action = 1
        _, reward, _, step_info = self.__sut.step(wait_action)

        logging.info("Checking step info for winning trades.")
        assert step_info['coeff'] == 1 + expected_coeff
        assert step_info['current_budget'] >= expected_budget_after_wait_at_least
        assert step_info['currently_invested'] == expected_invested_after_wait
        assert round(reward, 0) == round(expected_reward, 0)

    def test_traiding_environment_step__scenario_losing_trades(self) -> None:
        """
        Tests TradingEnvironment's step function - scenario losing trades.

        Verifies that reward is correctly calculated for losing trades, when
        no penalty is applied.

        Asserts:
            Coefficient is correctly calculated. The environment is correctly
            updated after action and correct reward is returned.
        """

        logging.info("Starting step scenario test case.")
        orders = [Order(200, False, 0.95, 1.05), Order(200, False, 0.95, 1.05)]
        self.__update_sut(leverage = 10,
                          current_orders = orders,
                          currently_placed_trades = 2,
                          current_budget = 600,
                          currently_invested = 400)

        expected_budget_after_wait_at_most = 1000
        expected_invested_after_wait = 0
        expected_coeff = (20800.0 - 20400.0) / 20400.0
        expected_orders_decrease = [-order.initial_value * expected_coeff for order in orders]
        expected_reward = np.sum(expected_orders_decrease) * self.__sut.get_broker().get_leverage()

        logging.info("Performing wait action.")
        wait_action = 1
        _, reward, _, step_info = self.__sut.step(wait_action)

        logging.info("Checking step info for losing trades.")
        assert step_info['coeff'] == 1 + expected_coeff
        assert step_info['current_budget'] <= expected_budget_after_wait_at_most
        assert step_info['currently_invested'] == expected_invested_after_wait
        assert round(reward, 0) == round(expected_reward, 0)

    def test_traiding_environment_step__scenario_getting_penalty(self) -> None:
        """
        Tests TradingEnvironment's step function - scenario getting penalty.

        Verifies that penalty is correctly calculated for winning trade after
        penalty starts to be applied, for no closed trades when penalty stops
        to be applied, for losing trades when negative rewards adjustment is
        applied.

        Asserts:
            For each step peanlty is correctly applied - inflience on reward.
            First action - reward is a fraction of full positive reward. Second
            action - there is no reward. Third step - negative reward minus
            negative reward adjustment. The environment is correctly updated
            after each action and correct reward is returned.
        """

        logging.info("Starting step scenario test case.")
        traiding_consts = self.__sut.get_trading_consts()
        orders = [Order(200, True, 0.9, 1.1), Order(200, False, 0.8, 1.2)]
        self.__update_sut(leverage = 10,
                          current_orders = orders,
                          currently_placed_long_trades = 1,
                          currently_placed_short_trades = 1,
                          current_budget = 600,
                          currently_invested = 400,
                          no_trades_placed_for = traiding_consts.PENALTY_STARTS)

        expected_budget_after_first_wait_at_least = 800
        expected_budget_after_third_wait_at_most = 1000
        expected_invested_after_first_wait = 200
        expected_invested_after_third_wait = 0
        expected_nr_of_long_trades_after_first_wait = 0
        expected_nr_of_short_trades_after_first_wait = 1
        expected_nr_of_long_trades_after_second_wait = 0
        expected_nr_of_short_trades_after_second_wait = 1
        expected_nr_of_long_trades_after_third_wait = 0
        expected_nr_of_short_trades_after_third_wait = 0
        expected_first_coeff = (20800.0 - 20400.0) / 20400.0
        expected_reward_after_first_wait_at_most = expected_first_coeff * \
            orders[0].initial_value * self.__sut.get_broker().get_leverage()
        expected_reward_after_second_wait = 0
        expected_reward_after_third_wait_at_most = -traiding_consts.STATIC_REWARD_ADJUSTMENT

        logging.info("Performing wait action.")
        wait_action = 1
        _, reward, _, step_info = self.__sut.step(wait_action)
        # no_trades_placed_for = PENALTY_STARTS + 1

        logging.info("Checking step info for getting penalty.")
        assert step_info['coeff'] == 1 + expected_first_coeff
        assert step_info['current_budget'] >= expected_budget_after_first_wait_at_least
        assert step_info['currently_invested'] == expected_invested_after_first_wait
        assert step_info['currently_placed_long_trades'] == expected_nr_of_long_trades_after_first_wait
        assert step_info['currently_placed_short_trades'] == expected_nr_of_short_trades_after_first_wait
        assert round(reward, 0) <= expected_reward_after_first_wait_at_most

        logging.info("Performing wait action.")
        _, reward, _, step_info = self.__sut.step(wait_action)
        # no_trades_placed_for = PENALTY_STOPS

        logging.info("Checking step info for getting penalty.")
        assert step_info['currently_placed_long_trades'] == expected_nr_of_long_trades_after_second_wait
        assert step_info['currently_placed_short_trades'] == expected_nr_of_short_trades_after_second_wait
        assert round(reward, 0) == expected_reward_after_second_wait

        logging.info("Performing wait action.")
        _, reward, _, step_info = self.__sut.step(wait_action)
        # no_trades_placed_for = PENALTY_STOPS + 1

        logging.info("Checking step info for getting penalty.")
        assert step_info['current_budget'] <= expected_budget_after_third_wait_at_most
        assert step_info['currently_invested'] == expected_invested_after_third_wait
        assert step_info['currently_placed_long_trades'] == expected_nr_of_long_trades_after_third_wait
        assert step_info['currently_placed_short_trades'] == expected_nr_of_short_trades_after_third_wait
        assert round(reward, 0) <= expected_reward_after_third_wait_at_most

    def test_traiding_environment_reset(self) -> None:
        """
        Tests TradingEnvironment's reset function.

        Verifies trading data is correctly cleared and reset.

        Asserts:
            Trading data equals either 0 or it's initial value.
        """

        logging.info("Starting reset test case.")
        traiding_consts = self.__sut.get_trading_consts()
        expected_budget = traiding_consts.INITIAL_BUDGET
        expected_invested = 0
        expected_no_trades_for = 0
        expected_nr_of_long_trades = 0
        expected_nr_of_short_trades = 0
        expected_orders = []

        logging.info("Performing reset.")
        self.__sut.reset()

        logging.info("Checking reset results.")
        traiding_data = self.__sut.get_trading_data()
        orders = self.__sut.get_broker().get_current_orders()
        assert traiding_data.current_budget == expected_budget
        assert traiding_data.currently_invested == expected_invested
        assert traiding_data.no_trades_placed_for == expected_no_trades_for
        assert traiding_data.currently_placed_long_trades == expected_nr_of_long_trades
        assert traiding_data.currently_placed_short_trades == expected_nr_of_short_trades
        assert orders == expected_orders

    def test_traiding_environment_get_labeled_data(self) -> None:
        """
        Tests the get_labeled_data method of the TradingEnvironment.

        Verifies that the method correctly retrieves and processes labeled data,
        including normalization and balancing of input and output data.

        Asserts:
            The input data is normalized correctly based on the mock data.
            The output data is balanced and matches the expected values.
            No test data is returned since should_split is False.
        """

        logging.info("Starting get labeled data test case.")
        self.__mocked_label_annotator.annotate.return_value = pd.Series(np.array([1, 0, 1, 0, None]))
        self.__mocked_data_balancer.balance.side_effect = lambda input_data, output_data: (input_data, output_data)
        expected_input_data = [
            [   # first data point
                -1, # normalized low[0] -> lower value from 20000.0 and 20500.0
                0,  # normalized high[0] -> equal value from 20900.0 and 20900.0
                -1, # normalized open[0] -> lower value from 20050.0 and 20600.0
                1,  # normalized close[0] -> higher value from 20600.0 and 20400.0
                -1, # normalized volume[0] -> lower value from 1000.0 and 1200.0
                1,  # normalized low[1] -> higher value from 20000.0 and 20500.0
                0,  # normalized high[1] -> equal value from 20900.0 and 20900.0
                1,  # normalized open[1] -> higher value from 20050.0 and 20600.0
                -1, # normalized close[1] -> lower value from 20600.0 and 20400.0
                1   # normalized volume[1] -> higher value from 1000.0 and 1200.0
            ],
            [   # second data point
                1, # normalized low[1] -> higher value from 20500.0 and 20100.0
                -1,  # normalized high[1] -> lower value from 20900.0 and 21000.0
                1, # normalized open[1] -> higher value from 20600.0 and 20400.0
                -1,  # normalized close[1] -> lower value from 20400.0 and 20800.0
                1, # normalized volume[1] -> higher value from 1200.0 and 1100.0
                -1,  # normalized low[2] -> lower value from 20500.0 and 20100.0
                1,  # normalized high[2] -> higher value from 20900.0 and 21000.0
                -1,  # normalized open[2] -> lower value from 20600.0 and 20400.0
                1, # normalized close[2] -> higher value from 20400.0 and 20800.0
                -1   # normalized volume[2] -> lower value from 1200.0 and 1100.0
            ]
        ]
        expected_output_data = [1, 0]

        logging.info("Performing get labeled data.")
        input_data, output_data, input_data_test, output_data_test = self.__sut.get_labeled_data(should_split = False)

        logging.info("Checking labeled data results.")
        self.__mocked_data_balancer.balance.assert_called_once()
        self.assertEqual(input_data.tolist(), expected_input_data)
        self.assertEqual(output_data.tolist(), expected_output_data)
        self.assertEqual(input_data_test.tolist(), [])
        self.assertEqual(output_data_test.tolist(), [])
