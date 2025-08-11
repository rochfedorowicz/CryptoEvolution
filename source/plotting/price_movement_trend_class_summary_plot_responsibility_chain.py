# plotting/price_movement_trend_class_summary_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# local imports
from source.agent import ClassificationLearningStrategyHandler
from source.plotting import PlotResponsibilityChainBase

class PriceMovementTrendClassSummaryPlotResponsibilityChain(PlotResponsibilityChainBase):
    """
    Implements a plotting responsibility chain for price movement trend classification results.
    """

    def _can_plot(self, key: str) -> bool:
        """
        Checks if the plot can be generated for the given key.

        Parameters:
            key (str): The key to check.

        Returns:
            (bool): True if the plot can be generated, False otherwise.
        """

        return key == ClassificationLearningStrategyHandler.PLOTTING_KEYS[0]

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates the plot for the price movement trend classification results.

        Parameters:
            plot_data (dict): The data to be plotted.

        Returns:
            (plt.Axes): The axes object containing the plot.
        """

        train_part_labels = plot_data.get("train_part_labels", None)
        train_part_price_movement = plot_data.get("train_part_price_movement", None)
        test_part_labels = plot_data.get("test_part_labels", None)
        test_part_price_movement = plot_data.get("test_part_price_movement", None)

        if train_part_labels is None or test_part_labels is None or \
           train_part_price_movement is None or test_part_price_movement is None:
            logging.warning(f"Insufficient data for plotting results under key: {ClassificationLearningStrategyHandler.PLOTTING_KEYS[0]}.")
            plt.text(0.5, 0.5, "Insufficient data for plotting",
                     ha = 'center', va = 'center', fontsize = 12)
            return plt.gca()

        fig = plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        gs = GridSpec(2, 1, figure = fig)

        # Plot 1: Price movements classified by trend
        plt.subplot(gs[0, 0])
        plt.title("Price changes by trend classification")

        # Temporarily assume there is only one way to classify trends
        trend_colors = ['green', (0.5, 0.5, 0.5, 0.2), 'red']
        class_names = ['UP Trend', 'NO Trend', 'DOWN Trend']
        price_movement_to_label_diff = len(train_part_price_movement) - len(train_part_labels) - 1

        for i in range(price_movement_to_label_diff + 1, len(train_part_labels)):
            color = trend_colors[train_part_labels[i - 1 - price_movement_to_label_diff]]
            plt.plot([i - 1, i], [train_part_price_movement[i - 1], train_part_price_movement[i]],
                     color = color, linewidth = 1)
        plt.axvline(x = len(train_part_labels) + price_movement_to_label_diff / 2,
                    color = 'black', linestyle = '--', linewidth = 2)
        for i in range(price_movement_to_label_diff + 1, len(test_part_labels)):
            color = trend_colors[test_part_labels[i - 1 - price_movement_to_label_diff]]
            plt.plot([i - 1 + len(train_part_labels), i + len(train_part_labels)],
                     [test_part_price_movement[i - 1], test_part_price_movement[i]],
                     color = color, linewidth = 1)

        legend_elements = [
            Line2D([0], [0], color = 'green', lw = 2, label = class_names[0]),
            Line2D([0], [0], color = 'gray', lw = 2, label = class_names[1]),
            Line2D([0], [0], color = 'red', lw = 2, label = class_names[2])
        ]
        plt.legend(handles = legend_elements, loc = 'upper left')

        plt.yscale('log')
        plt.xlabel('Trading points')
        plt.ylabel('Log scale of price')

        # Plot 2: Class distribution
        plt.subplot(gs[1, 0])
        plt.title("Class distribution of price movements")

        unique_train_labels, train_counts = np.unique(train_part_labels, return_counts = True)
        x_ticks = np.arange(len(unique_train_labels))
        unique_test_labels, test_counts = np.unique(test_part_labels, return_counts = True)
        shift = 0.2

        train_bars = plt.bar(x_ticks - shift, train_counts, label = 'Train', width = 0.4,
                             color = [trend_colors[int(label)] for label in unique_train_labels])
        test_bars = plt.bar(x_ticks + shift, test_counts, label = 'Test', width = 0.4, alpha = 0.7,
                            color = [trend_colors[int(label)] for label in unique_test_labels])

        plt.xticks(x_ticks, [class_names[int(label)] for label in unique_train_labels])

        for bar, count in zip(train_bars, train_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha = 'center', va = 'bottom', fontsize = 9)

        for bar, count in zip(test_bars, test_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha = 'center', va = 'bottom', fontsize = 9)

        plt.xlabel('Price movement class')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()

        return plt.gca()