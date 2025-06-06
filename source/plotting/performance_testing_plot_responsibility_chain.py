# plotting/performance_testing_plot_responsibility_chain.py

# global imports
import matplotlib.pyplot as plt
import numpy as np

# local imports
from source.agent import PerformanceTestingStrategyHandler
from source.plotting import PlotResponsibilityChainBase

class PerformanceTestingPlotResponsibilityChain(PlotResponsibilityChainBase):
    """
    Implements a plotting responsibility chain for performance testing results.
    It implements the _can_plot and _plot methods to visualize assets values, currency prices,
    and solvency coefficients over time.
    """

    def __init__(self, window_size: int = 5) -> None:
        """
        Class constructor. Initializes the plotting responsibility chain with a window size.

        Parameters:
            window_size (int): The size of the moving average window for smoothing the assets values.
        """

        self.__window_size = window_size

    def _can_plot(self, key: str) -> bool:
        """
        Checks if the plot can be generated for the given key.

        Parameters:
            key (str): The key to check.

        Returns:
            (bool): True if the plot can be generated, False otherwise.
        """

        return key == PerformanceTestingStrategyHandler.PLOTTING_KEY

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates the plot for the performance testing results.

        Parameters:
            plot_data (dict): The data to be plotted.

        Returns:
            (plt.Axes): The axes object containing the plot.
        """

        assets_values = plot_data['assets_values']
        currency_prices = plot_data['currency_prices']
        steps = plot_data['iterations']
        solvency_coefficient = plot_data['solvency_coefficient']
        adjusted_window_size = min(self.__window_size, len(assets_values))
        filter = np.ones(adjusted_window_size) / adjusted_window_size
        avg_assets_values = np.convolve(assets_values, filter, mode = 'same')

        plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        plt.plot(range(steps[0], steps[0] + len(currency_prices)), currency_prices, label = 'Currency price growth', color = 'green')
        plt.plot(steps, assets_values, label = 'Assets value growth', color = 'blue')
        plt.plot(steps, avg_assets_values,
                 label = f'{adjusted_window_size}-step moving average of assets value growth', color = 'red')
        plt.title(f'Testing history with solvency {solvency_coefficient}')
        plt.xlabel('Number of steps')
        plt.ylabel('Currency price and assets value growth')
        plt.legend()

        return plt.gca()
