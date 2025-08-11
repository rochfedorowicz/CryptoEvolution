# plotting/asset_price_movement_summary_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# local imports
from source.agent import LearningStrategyHandlerBase
from source.plotting import PlotResponsibilityChainBase

class AssetPriceMovementSummaryPlotResponsibilityChain(PlotResponsibilityChainBase):
    """
    Implements a plotting responsibility chain for asset price movement summary plots.
    It implements the _can_plot and _plot methods to visualize asset price movements
    and volatility over time.
    """

    def _can_plot(self, key: str) -> bool:
        """
        Checks if the plot can be generated for the given key.

        Parameters:
            key (str): The key to check.

        Returns:
            (bool): True if the plot can be generated, False otherwise.
        """

        return key == LearningStrategyHandlerBase.PLOTTING_KEY

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates the asset price movement summary plot based on the provided data.

        Parameters:
            plot_data (dict): The data to be plotted.

        Returns:
            (plt.Axes): The axes object containing the plot.
        """

        train_part_price_movement = plot_data.get("train_part_price_movement", None)
        train_part_volatility = plot_data.get("train_part_volatility", None)
        test_part_price_movement = plot_data.get("test_part_price_movement", None)
        test_part_volatility = plot_data.get("test_part_volatility", None)

        if train_part_price_movement is None or test_part_price_movement is None or \
           train_part_volatility is None or test_part_volatility is None:
            logging.warning(f"Insufficient data for plotting results under key: {LearningStrategyHandlerBase.PLOTTING_KEY}.")
            plt.text(0.5, 0.5, "Insufficient data for plotting",
                     ha = 'center', va = 'center', fontsize = 12)
            return plt.gca()

        fig = plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        gs = GridSpec(3, 1, figure = fig)

        # Plot 1: Currency prices
        plt.subplot(gs[0, 0])
        plt.title("Currency Prices")

        train_x = range(len(train_part_price_movement))
        plt.plot(train_x, train_part_price_movement, label = 'Train Prices')

        test_x = range(len(train_part_price_movement),
                       len(train_part_price_movement) + len(test_part_price_movement))
        plt.plot(test_x, test_part_price_movement, label = 'Test Prices')

        plt.axvline(x = len(train_part_price_movement) - 1, color = 'r',
                    linestyle = '--', alpha = 0.7, label = 'Train/Test Split')

        plt.yscale('log')
        plt.xlabel('Trading points')
        plt.ylabel('Log scale of price')
        plt.legend(loc = 'upper left')

        # Plot 2: Volatility
        plt.subplot(gs[1, 0])
        plt.title("Volatility")

        plt.plot(train_x, train_part_volatility, label='Train Volatility')
        plt.plot(test_x, test_part_volatility, label='Test Volatility')
        plt.xlabel('Trading points')
        plt.ylabel('Volatility')
        plt.legend(loc='upper left')

        # Plot 3: Price changes with volatility
        plt.subplot(gs[2, 0])
        plt.title("Price Changes with Volatility")
        colormap = cm.get_cmap('plasma')

        train_vol_normalized = np.array(train_part_volatility) / max(train_part_volatility)
        for i in range(1, len(train_x)):
            plt.plot([train_x[i-1], train_x[i]],
                    [train_part_price_movement[i-1], train_part_price_movement[i]],
                    color = colormap(train_vol_normalized[i]),
                    linewidth = 1)

        test_vol_normalized = np.array(test_part_volatility) / max(test_part_volatility)
        for i in range(1, len(test_x)):
            plt.plot([test_x[i-1], test_x[i]],
                    [test_part_price_movement[i-1], test_part_price_movement[i]],
                    color = colormap(test_vol_normalized[i]),
                    linewidth = 1)

        norm = colors.Normalize(vmin = 0, vmax = max(np.array(train_part_volatility).max(),
                                                     np.array(test_part_volatility).max()))
        sm = cm.ScalarMappable(cmap = colormap, norm = norm)
        sm.set_array([])

        plt.colorbar(sm, ax = plt.gca())
        plt.yscale('log')
        plt.xlabel('Trading points')
        plt.ylabel('Log scale of price')
        plt.title('Price with Volatility-Colored Line')
        plt.tight_layout()

        return plt.gca()