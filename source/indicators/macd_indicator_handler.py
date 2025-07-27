# indicators/macd_indicator_handler.py

# global imports
import pandas as pd

# local imports
from source.indicators import ExponentialMovingAverageIndicatorHandler, IndicatorHandlerBase

class MovingAverageConvergenceDivergenceIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Moving Average Convergence Divergence (MACD) indicator. It shows the relationship
    between two moving averages of an asset's price.
    """

    def __init__(self, fast_period_window_size: int = 12, slow_period_window_size: int = 26,
        signal_period_window_size: int = 9) -> None:
        """
        Class constructor.

        Parameters:
            fast_period_window_size (int): Length of the fast EMA window.
            slow_period_window_size (int): Length of the slow EMA window.
            signal_period_window_size (int): Length of the signal line window.
        """

        self.__fast_period_window_size = fast_period_window_size
        self.__slow_period_window_size = slow_period_window_size
        self.__signal_period_window_size = signal_period_window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Moving Average Convergence Divergence (MACD) indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated MACD values.
        """

        fast_ema = ExponentialMovingAverageIndicatorHandler(self.__fast_period_window_size).calculate(data)
        slow_ema = ExponentialMovingAverageIndicatorHandler(self.__slow_period_window_size).calculate(data)

        macd_data_df = pd.DataFrame(index = data.index)
        macd_data_df['macd_line'] = fast_ema['ema'] - slow_ema['ema']
        macd_data_df['macd_sig'] = macd_data_df['macd_line']. \
            ewm(span = self.__signal_period_window_size, adjust = False).mean()
        macd_data_df['macd_hist'] = macd_data_df['macd_line'] - macd_data_df['macd_sig']

        return macd_data_df

    def can_be_normalized(self) -> bool:
        """
        Checks if the indicator can be normalized.

        Returns:
            (bool): True if the indicator can be normalized, False otherwise.
        """

        return True