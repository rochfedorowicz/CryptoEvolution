# indicators/average_true_range_indicator_handler.py

# global imports
import pandas as pd

# local imports
from source.indicators import IndicatorHandlerBase

class AverageTrueRangeIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Average True Range (ATR) indicator. It measures market volatility by decomposing
    the entire range of an asset price for that period.
    """

    def __init__(self, window_size: int = 14) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Length of window that indicator should be applied over.
        """

        self.__window_size = window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Average True Range (ATR) indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated ATR values.
        """

        prev_close = data['close'].shift()
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - prev_close)
        low_close_prev = abs(data['low'] - prev_close)
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis = 1).max(axis = 1)

        atr_df = pd.DataFrame(index = data.index)
        atr_df['atr'] = true_range.rolling(window = self.__window_size, min_periods = 1).mean()

        return atr_df

    def can_be_normalized(self) -> bool:
        """
        Checks if the indicator can be normalized.

        Returns:
            (bool): True if the indicator can be normalized, False otherwise.
        """

        return True