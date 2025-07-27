# indicators/ema_indicator_handler.py

# global imports
import pandas as pd

# local imports
from source.indicators import IndicatorHandlerBase

class ExponentialMovingAverageIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Exponential Moving Average (EMA) indicator. It gives more weight to recent prices
    and reacts more quickly to price changes than a simple moving average.
    """

    def __init__(self, window_size: int = 20) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Length of window that indicator should be applied over.
        """

        self.__window_size = window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Exponential Moving Average (EMA) indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated EMA values.
        """

        ema_data_df = pd.DataFrame(index = data.index)
        ema_data_df['ema'] = data['close'].ewm(span = self.__window_size, adjust = False).mean()

        return ema_data_df

    def can_be_normalized(self) -> bool:
        """
        Checks if the indicator can be normalized.

        Returns:
            (bool): True if the indicator can be normalized, False otherwise.
        """

        return True