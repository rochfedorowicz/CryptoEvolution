# plotting/performance_testing_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# local imports
from source.agent import PerformanceTestingStrategyHandler
from source.plotting import PlotResponsibilityChainBase

class PerformanceTestingPlotResponsibilityChain(PlotResponsibilityChainBase):
    """
    Implements a plotting responsibility chain for performance testing results.
    It implements the _can_plot and _plot methods to visualize assets values, currency prices,
    and solvency coefficients over time.
    """

    def __init__(self, window_size: int = 30, risk_free_rate: float = 0.042) -> None:
        """
        Class constructor. Initializes the plotting responsibility chain with a window size.

        Parameters:
            window_size (int): The size of the moving average window for smoothing the assets values.
            risk_free_rate (float): The risk-free rate to be used in the Sharpe ratio calculation.
        """

        self.__window_size = window_size
        self.__risk_free_rate = risk_free_rate

    def _can_plot(self, key: str) -> bool:
        """
        Checks if the plot can be generated for the given key.

        Parameters:
            key (str): The key to check.

        Returns:
            (bool): True if the plot can be generated, False otherwise.
        """

        return key == PerformanceTestingStrategyHandler.PLOTTING_KEY

    def __calculate_sharpe_ratio_for_period(self, input_values: list[float], trading_point_risk_free_rate: float,
        annualization_factor: float) -> float:
        """
        Calculates the Sharpe ratio for a given period.

        Parameters:
            input_values (list[float]): The input values for the period.
            trading_point_risk_free_rate (float): The risk-free rate to use for the calculation.
            annualization_factor (float): The factor to annualize the Sharpe ratio.

        Returns:
            (float): The calculated Sharpe ratio.
        """

        returns = pd.Series(input_values).pct_change().fillna(0) - trading_point_risk_free_rate
        period_std_return = returns.std()

        if period_std_return > 0:
            sharpe_ratio = returns.mean() / period_std_return * annualization_factor
            sharpe_ratio = np.clip(sharpe_ratio, -2, 3)
        else:
            sharpe_ratio = 0.0

        return sharpe_ratio

    def __calculate_sharpe_ratio(self, input_values: list[float], trading_point_risk_free_rate: float,
        annualization_factor: float, sharpe_ratio_window_size: int) -> list[float]:
        """
        Calculates the rolling Sharpe ratio for the given input values.

        Parameters:
            input_values (list[float]): The input values for the period.
            trading_point_risk_free_rate (float): The risk-free rate to use for the calculation.
            annualization_factor (float): The factor to annualize the Sharpe ratio.
            sharpe_ratio_window_size (int): The window size for the rolling Sharpe ratio.

        Returns:
            (list[float]): The rolling Sharpe ratio values.
        """

        rolling_sharpe_ratio = []
        last_sharpe_ratio = 0
        for i in range(len(input_values)):
            if i % sharpe_ratio_window_size == 0:
                start_idx = max(0, i - sharpe_ratio_window_size + 1)
                last_sharpe_ratio = self.__calculate_sharpe_ratio_for_period(input_values[start_idx:i + 1],
                                                                             trading_point_risk_free_rate,
                                                                             annualization_factor)
            rolling_sharpe_ratio.append(last_sharpe_ratio)

        return rolling_sharpe_ratio

    def __calculate_annual_return(self, input_values: list[float], trading_points_per_year: int) -> list[float]:
        """
        Calculates the rolling annual return for the given input values.

        Parameters:
            input_values (list[float]): The input values for the period.
            trading_points_per_year (int): The number of trading points per year.

        Returns:
            (list[float]): The rolling annual return values.
        """

        rolling_annual_return = []
        for i in range(len(input_values)):
            start_idx = max(0, i - trading_points_per_year + 1)
            annual_return_pct = ((input_values[i] / input_values[start_idx]) - 1) * 100.0
            annual_return_pct = np.clip(annual_return_pct, -100.0, 1000.0)
            rolling_annual_return.append(annual_return_pct)

        return rolling_annual_return

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates the plot for the performance testing results.

        Parameters:
            plot_data (dict): The data to be plotted.

        Returns:
            (plt.Axes): The axes object containing the plot.
        """

        assets_values = plot_data.get("assets_values", None)
        currency_prices = plot_data.get("currency_prices", None)
        iterations = plot_data.get("iterations", None)
        solvency_coefficient = plot_data.get("solvency_coefficient", None)
        trading_points_per_year = plot_data.get("trading_points_per_year", None)

        if assets_values is None or currency_prices is None or iterations is None \
            or solvency_coefficient is None or trading_points_per_year is None:
            logging.warning(f"Insufficient data for plotting results under key: {PerformanceTestingStrategyHandler.PLOTTING_KEY}.")
            plt.text(0.5, 0.5, "Insufficient data for plotting",
                     ha = 'center', va = 'center', fontsize = 12)
            return plt.gca()

        adjusted_window_size = min(self.__window_size, len(assets_values))

        fig = plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        gs = GridSpec(3, 1, figure = fig)

        # Plot 1: Testing history
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title(f'Testing history with solvency {solvency_coefficient}')
        ax1.plot(iterations, currency_prices, label = 'Currency price growth', color = 'green')
        ax1.plot(iterations, assets_values, label = 'Assets value growth', color = 'blue')

        avg_assets_values = pd.Series(assets_values).rolling(
            window = adjusted_window_size,
            center = True,
            min_periods = 1
        ).mean().values

        ax1.plot(iterations, avg_assets_values, label = f'{adjusted_window_size}-step moving average of assets value growth', color = 'red')
        ax1.set_xlabel('Number of steps')
        ax1.set_ylabel('Normalized asset growth')
        ax1.legend(loc = 'upper left')

        # Plot 2: Rolling Sharpe Ratio
        ax2 = plt.subplot(gs[1, 0])
        ax2.set_title(f'Annualized Rolling Sharpe Ratio (risk-free rate = {100 * self.__risk_free_rate}%)')
        ax2.axhline(y = 0, color = 'red', linestyle = '-', alpha = 0.3)
        ax2.axhline(y = 1.0, color = 'orange', linestyle = '--', alpha = 0.5)
        ax2.axhline(y = 2.0, color = 'green', linestyle = '--', alpha = 0.5)

        trading_point_risk_free_rate = self.__risk_free_rate / trading_points_per_year
        sharpe_ratio_window_size = int(trading_points_per_year / 12)
        annualization_factor = np.sqrt(trading_points_per_year)
        asset_rolling_sharpe_ratio = self.__calculate_sharpe_ratio(assets_values, trading_point_risk_free_rate,
                                                                   annualization_factor, sharpe_ratio_window_size)
        price_rolling_sharpe_ratio = self.__calculate_sharpe_ratio(currency_prices, trading_point_risk_free_rate,
                                                                   annualization_factor, sharpe_ratio_window_size)

        total_assets_sharpe_ratio = self.__calculate_sharpe_ratio_for_period(assets_values, trading_point_risk_free_rate, annualization_factor)
        total_price_sharpe_ratio = self.__calculate_sharpe_ratio_for_period(currency_prices, trading_point_risk_free_rate, annualization_factor)
        asset_label = f'{sharpe_ratio_window_size}-step rolling Sharpe ratio for assets, total Sharpe Ratio = {total_assets_sharpe_ratio:.2f}'
        price_label = f'{sharpe_ratio_window_size}-step rolling Sharpe ratio for holding currency, total Sharpe Ratio = {total_price_sharpe_ratio:.2f}'

        ax2.plot(iterations, asset_rolling_sharpe_ratio, label = asset_label, color = 'purple', linewidth = 3)
        ax2.plot(iterations, price_rolling_sharpe_ratio, label = price_label, color = 'red')
        ax2.set_xlabel('Number of steps')
        ax2.set_ylabel('Sharpe Ratio value')
        ax2.set_ylim(max(-2.0, min(min(asset_rolling_sharpe_ratio), min(price_rolling_sharpe_ratio))) - 0.1,
                     max(2.0, max(asset_rolling_sharpe_ratio), max(price_rolling_sharpe_ratio)) + 2)
        ax2.legend(loc = 'upper left')

        # Plot 3: Rolling Annual Return
        ax3 = plt.subplot(gs[2, 0])
        ax3.set_title(f'Rolling Annual Return (year length = {trading_points_per_year} trading points)')

        asset_rolling_annual_return = self.__calculate_annual_return(assets_values, trading_points_per_year)
        price_rolling_annual_return = self.__calculate_annual_return(currency_prices, trading_points_per_year)
        total_asset_return = ((assets_values[-1] / assets_values[0]) - 1) * 100.0
        total_price_return = ((currency_prices[-1] / currency_prices[0]) - 1) * 100.0
        asset_label = f'{trading_points_per_year}-step rolling annual return for assets, total return = {total_asset_return:.2f}%'
        price_label = f'{trading_points_per_year}-step rolling annual return for holding currency, total return = {total_price_return:.2f}%'

        ax3.plot(iterations, asset_rolling_annual_return, label = asset_label, color = 'orange')
        ax3.plot(iterations, price_rolling_annual_return, label = price_label, color = 'green')
        ax3.set_xlabel('Number of steps')
        ax3.set_ylabel('Annual Return (%)')
        ax3.legend(loc = 'upper left')
        plt.tight_layout()

        return plt.gca()
