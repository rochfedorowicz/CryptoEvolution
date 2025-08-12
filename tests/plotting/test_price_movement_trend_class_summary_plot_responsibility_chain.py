# tests/plotting/test_price_movement_trend_class_summary_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.patches
import matplotlib.pyplot as plt
from ddt import ddt
from unittest import TestCase

# local imports
from source.agent import ClassificationLearningStrategyHandler
from source.plotting import PriceMovementTrendClassSummaryPlotResponsibilityChain

@ddt
class PriceMovementTrendClassSummaryPlotResponsibilityChainTestCase(TestCase):
    """
    Test case for PriceMovementTrendClassSummaryPlotResponsibilityChain class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: PriceMovementTrendClassSummaryPlotResponsibilityChain = PriceMovementTrendClassSummaryPlotResponsibilityChain()

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")
        plt.close('all')

    def test_price_movement_trend_class_summary_plot_responsibility_chain_plot__able_to_handle(self) -> None:
        """
        Tests PriceMovementTrendClassSummaryPlotResponsibilityChain's plot functionality with valid input.

        Verifies that the plot method correctly handles input data with the 'price_movement_trend_class_summary'
        key and generates a plot with the expected title, labels, and data series. Tests that
        the plot contains three lines representing currency prices, asset values, and the
        moving average.

        Asserts:
            The plot has the correct title, axis labels, and number of data series.
            The data points in each plot match the input data.
        """

        mocked_input_data = {
            'key': ClassificationLearningStrategyHandler.PLOTTING_KEYS[0],
            'plot_data': {
                'train_part_labels': [0, 1, 0, 1, 2],
                'train_part_price_movement': [80000.0, 80010.0, 80000.0, 80030.0, 80020.0],
                'test_part_labels': [1, 0, 1, 2, 0],
                'test_part_price_movement': [80050.0, 80030.0, 80070.0, 80040.0, 80040.0]
            }
        }

        # Plot 1: Price movements classified by trend
        expected_price_movements_ax_title = 'Price changes by trend classification'
        expected_price_movements_ax_xlabel = 'Trading points'
        expected_price_movements_ax_ylabel = 'Log scale of price'
        train_part_price_movement = mocked_input_data['plot_data']['train_part_price_movement']
        test_part_price_movement = mocked_input_data['plot_data']['test_part_price_movement']
        expected_price_movements_ax_number_of_lines = \
            len(train_part_price_movement) + len(test_part_price_movement) + 1

        # Plot 2: Class distribution
        expected_class_distribution_ax_title = 'Class distribution of price movements'
        expected_class_distribution_ax_xlabel = 'Price movement class'
        expected_class_distribution_ax_ylabel = 'Frequency'
        expected_class_distribution_ax_bar_heights = [2, 2, 1, 2, 2, 1]
        expected_class_distribution_ax_legend_texts = ['Train', 'Test']

        logging.info("Plotting using provided data.")
        _ = self.__sut.plot(mocked_input_data)
        fig = plt.gcf()

        # Plot 1: Price movements classified by trend
        price_movements_ax = fig.axes[0]
        price_movement_lines = price_movements_ax.get_lines()

        # Plot 2: Class distribution
        class_distribution_ax = fig.axes[1]
        drawn_bars_class_distribution = \
            [child for child in class_distribution_ax.get_children()[:-1] \
             if isinstance(child, matplotlib.patches.Rectangle)]
        legend_texts_class_distribution = \
            [text.get_text() for text in class_distribution_ax.get_legend().get_texts()]

        logging.info("Checking expected plots.")
        self.assertEqual(len(fig.axes), 2)

        # Plot 1: Price movements classified by trend
        self.assertEqual(price_movements_ax.get_title(), expected_price_movements_ax_title)
        self.assertEqual(price_movements_ax.get_xlabel(), expected_price_movements_ax_xlabel)
        self.assertEqual(price_movements_ax.get_ylabel(), expected_price_movements_ax_ylabel)
        self.assertEqual(len(price_movement_lines), expected_price_movements_ax_number_of_lines)

        # Plot 2: Class distribution
        self.assertEqual(class_distribution_ax.get_title(), expected_class_distribution_ax_title)
        self.assertEqual(class_distribution_ax.get_xlabel(), expected_class_distribution_ax_xlabel)
        self.assertEqual(class_distribution_ax.get_ylabel(), expected_class_distribution_ax_ylabel)
        self.assertEqual(len(drawn_bars_class_distribution), 6)
        self.assertEqual([bar.get_height() for bar in drawn_bars_class_distribution],
                         expected_class_distribution_ax_bar_heights)
        self.assertEqual(legend_texts_class_distribution, expected_class_distribution_ax_legend_texts)

    def test_price_movement_trend_class_summary_plot_responsibility_chain_plot__unable_to_handle(self) -> None:
        """
        Tests PriceMovementTrendClassSummaryPlotResponsibilityChain's plot functionality with invalid input.

        Verifies that the plot method correctly handles the case where the key does not
        match 'price_movement_trend_class_summary'. In this case, the handler should not process the request
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