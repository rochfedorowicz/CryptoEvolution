# tests/plotting/test_performance_testing_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
from ddt import ddt
from unittest import TestCase

# local imports
from source.agent import PerformanceTestingStrategyHandler
from source.plotting import PerformanceTestingPlotResponsibilityChain

@ddt
class PerformanceTestingPlotResponsibilityChainTestCase(TestCase):
    """
    Test case for PerformanceTestingPlotResponsibilityChain class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: PerformanceTestingPlotResponsibilityChain = PerformanceTestingPlotResponsibilityChain(3)

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")
        plt.close('all')

    def test_performance_testing_plot_responsibility_chain_plot__able_to_handle(self) -> None:
        """
        Tests PerformanceTestingPlotResponsibilityChain's plot functionality with valid input.

        Verifies that the plot method correctly handles input data with the 'performance_testing'
        key and generates a plot with the expected title, labels, and data series. Tests that
        the plot contains three lines representing currency prices, asset values, and the
        moving average.

        Asserts:
            The plot has the correct title, axis labels, and number of data series.
            The data points in each plot match the input data.
        """

        mocked_input_data = {
            'key': PerformanceTestingStrategyHandler.PLOTTING_KEY,
            'plot_data': {
                'assets_values': [80.0, 82.0, 82.0, 84.0, 88.0, 94.0, 102.0, 120.0, 112.0, 108.0],
                'currency_prices': [1000.0, 1005.0, 1005.0, 1010.0, 1020.0, 1035.0, 1055.0, 1100.0, 1080.0, 1070.0],
                'iterations': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                'solvency_coefficient': 1.4,
                'trading_points_per_year': 252
            }
        }

        expected_title = f"Testing history with solvency {mocked_input_data['plot_data']['solvency_coefficient']}"
        expected_xlabel = "Trading points"
        expected_ylabel = "Normalized asset growth"
        expected_number_of_plotted_lines = 3
        expected_xydata_first_line = [list(tup) for tup in list(zip(mocked_input_data['plot_data']['iterations'], mocked_input_data['plot_data']['currency_prices']))]
        expected_xydata_second_line = [list(tup) for tup in list(zip(mocked_input_data['plot_data']['iterations'], mocked_input_data['plot_data']['assets_values']))]

        logging.info("Plotting using provided data.")
        _ = self.__sut.plot(mocked_input_data)
        fig = plt.gcf()
        testing_history_ax = fig.axes[0]
        plotted_lines = testing_history_ax.get_lines()

        logging.info("Checking expected plot.")
        self.assertEqual(testing_history_ax.get_title(), expected_title)
        self.assertEqual(testing_history_ax.get_xlabel(), expected_xlabel)
        self.assertEqual(testing_history_ax.get_ylabel(), expected_ylabel)
        self.assertEqual(len(plotted_lines), expected_number_of_plotted_lines)
        self.assertEqual(plotted_lines[0].get_xydata().tolist(), expected_xydata_first_line)
        self.assertEqual(plotted_lines[1].get_xydata().tolist(), expected_xydata_second_line)

    def test_performance_testing_plot_responsibility_chain_plot__unable_to_handle(self) -> None:
        """
        Tests PerformanceTestingPlotResponsibilityChain's plot functionality with invalid input.

        Verifies that the plot method correctly handles the case where the key does not
        match 'performance_testing'. In this case, the handler should not process the request
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