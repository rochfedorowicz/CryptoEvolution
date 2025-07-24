# tests/plotting/test_asset_price_movement_summary_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
from ddt import ddt
from unittest import TestCase

# local imports
from source.agent import LearningStrategyHandlerBase
from source.plotting import AssetPriceMovementSummaryPlotResponsibilityChain

@ddt
class AssetPriceMovementSummaryPlotResponsibilityChainTestCase(TestCase):
    """
    Test case for AssetPriceMovementSummaryPlotResponsibilityChain class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: AssetPriceMovementSummaryPlotResponsibilityChain = AssetPriceMovementSummaryPlotResponsibilityChain()

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")
        plt.close('all')

    def test_asset_price_movement_summary_plot_responsibility_chain_plot__able_to_handle(self) -> None:
        """
        Tests AssetPriceMovementSummaryPlotResponsibilityChain's plot functionality with valid input.

        Verifies that the plot method correctly handles input data with the 'asset_price_movement_summary'
        key and generates a plot with the expected title, labels, and data series. Tests that
        the plot contains three lines representing currency prices, asset values, and the
        moving average.

        Asserts:
            The plot has the correct title, axis labels, and number of data series.
            The data points in each plot match the input data.
        """

        mocked_input_data = {
            'key': LearningStrategyHandlerBase.PLOTTING_KEY,
            'plot_data': {
                'train_part_price_movement': [80000.0, 80010.0, 80020.0, 80030.0, 80040.0],
                'train_part_volatility': [0.01, 0.02, 0.015, 0.025, 0.02],
                'test_part_price_movement': [80050.0, 80060.0, 80070.0, 80080.0, 80090.0],
                'test_part_volatility': [0.03, 0.04, 0.035, 0.045, 0.04]
            }
        }

        # Plot 1: Currency prices
        expected_currency_prices_ax_title = 'Currency Prices'
        expected_currency_prices_ax_xlabel = 'Time'
        expected_currency_prices_ax_ylabel = 'Log scale of price'
        expected_currency_prices_ax_number_of_lines = 3
        train_part_price_movement = mocked_input_data['plot_data']['train_part_price_movement']
        test_part_price_movement = mocked_input_data['plot_data']['test_part_price_movement']
        expected_currency_prices_ax_xydata_first_line = [list(tup) for tup in
            list(zip(range(len(train_part_price_movement)), train_part_price_movement))]
        expected_currency_prices_ax_xydata_second_line = [list(tup) for tup in
            list(zip(range(len(train_part_price_movement), len(train_part_price_movement) \
                           + len(test_part_price_movement)), test_part_price_movement))]

        # Plot 2: Volatility
        expected_volatility_ax_title = 'Volatility'
        expected_volatility_ax_xlabel = 'Time'
        expected_volatility_ax_ylabel = 'Volatility'
        expected_volatility_ax_number_of_lines = 2
        train_part_volatility = mocked_input_data['plot_data']['train_part_volatility']
        test_part_volatility = mocked_input_data['plot_data']['test_part_volatility']
        expected_volatility_ax_xydata_first_line = [list(tup) for tup in
            list(zip(range(len(train_part_volatility)), train_part_volatility))]
        expected_volatility_ax_xydata_second_line = [list(tup) for tup in
            list(zip(range(len(train_part_volatility), len(train_part_volatility) \
                           + len(test_part_volatility)), test_part_volatility))]

        # Plot 3: Price changes with volatility
        expected_currency_prices_with_volatility_ax_title = 'Price with Volatility-Colored Line'
        expected_currency_prices_with_volatility_ax_xlabel = 'Time'
        expected_currency_prices_with_volatility_ax_ylabel = 'Log scale of price'
        expected_currency_prices_with_volatility_ax_number_of_lines = \
            len(train_part_price_movement) - 1 + len(test_part_price_movement) - 1

        logging.info("Plotting using provided data.")
        _ = self.__sut.plot(mocked_input_data)
        fig = plt.gcf()

        # Plot 1: Currency prices
        currency_prices_ax = fig.axes[0]
        currency_prices_lines = currency_prices_ax.get_lines()

        # Plot 2: Volatility
        volatility_ax = fig.axes[1]
        volatility_lines = volatility_ax.get_lines()

        # Plot 3: Price changes with volatility
        currency_prices_with_volatility_ax = fig.axes[2]
        currency_prices_with_volatility_lines = currency_prices_with_volatility_ax.get_lines()

        logging.info("Checking expected plots.")
        self.assertEqual(len(fig.axes), 4)  # 3 main plots + 1 colorbar

        # Plot 1: Currency prices
        self.assertEqual(currency_prices_ax.get_title(), expected_currency_prices_ax_title)
        self.assertEqual(currency_prices_ax.get_xlabel(), expected_currency_prices_ax_xlabel)
        self.assertEqual(currency_prices_ax.get_ylabel(), expected_currency_prices_ax_ylabel)
        self.assertEqual(len(currency_prices_lines), expected_currency_prices_ax_number_of_lines)
        self.assertEqual(currency_prices_lines[0].get_xydata().tolist(),
                         expected_currency_prices_ax_xydata_first_line)
        self.assertEqual(currency_prices_lines[1].get_xydata().tolist(),
                         expected_currency_prices_ax_xydata_second_line)

        # Plot 2: Volatility
        self.assertEqual(volatility_ax.get_title(), expected_volatility_ax_title)
        self.assertEqual(volatility_ax.get_xlabel(), expected_volatility_ax_xlabel)
        self.assertEqual(volatility_ax.get_ylabel(), expected_volatility_ax_ylabel)
        self.assertEqual(len(volatility_lines), expected_volatility_ax_number_of_lines)
        self.assertEqual(volatility_lines[0].get_xydata().tolist(),
                         expected_volatility_ax_xydata_first_line)
        self.assertEqual(volatility_lines[1].get_xydata().tolist(),
                         expected_volatility_ax_xydata_second_line)

        # Plot 3: Price changes with volatility
        self.assertEqual(currency_prices_with_volatility_ax.get_title(),
                         expected_currency_prices_with_volatility_ax_title)
        self.assertEqual(currency_prices_with_volatility_ax.get_xlabel(),
                         expected_currency_prices_with_volatility_ax_xlabel)
        self.assertEqual(currency_prices_with_volatility_ax.get_ylabel(),
                         expected_currency_prices_with_volatility_ax_ylabel)
        self.assertEqual(len(currency_prices_with_volatility_lines),
                         expected_currency_prices_with_volatility_ax_number_of_lines)


    def test_asset_price_movement_summary_plot_responsibility_chain_plot__unable_to_handle(self) -> None:
        """
        Tests AssetPriceMovementSummaryPlotResponsibilityChain's plot functionality with invalid input.

        Verifies that the plot method correctly handles the case where the key does not
        match 'asset_price_movement_summary'. In this case, the handler should not process the request
        and should return None, indicating that the request should be passed to the next
        handler in the chain.

        Asserts:
            The method returns None when given an unrecognized key.
        """

        logging.info("Starting plot test.")
        mocked_input_data = {
            'key': 'unknown_plot_type',
            'plot_data': None
        }

        logging.info("Plotting using provided data.")
        result = self.__sut.plot(mocked_input_data)

        logging.info("Checking if plot is empty.")
        self.assertEqual(result, None)