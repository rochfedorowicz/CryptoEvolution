# indicators/volatility_indicator_handler.py

# global imports
import pandas as pd

# local imports
from source.indicators import IndicatorHandlerBase

class VolatilityIndicatorHandler(IndicatorHandlerBase):
    """
    Implements volatility indicator. It measures the price fluctuations over a
    certain period of time.
    """

    def __init__(self, window_size: int = 10) -> None:
        """
        Class constructor. Initializes the window size for volatility calculation.

        Parameters:
            window_size (int): Size of the window used for calculating volatility.
        """

        self.__window_size = window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates volatility indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated volatility values.
        """

        volatility_df = pd.DataFrame(index = data.index)
        volatility_df['volatility'] = data['close'].pct_change(). \
            rolling(window = self.__window_size, min_periods = 1).std()
        volatility_df = volatility_df.fillna(0)

        return volatility_df

    def can_be_normalized(self) -> bool:
        """
        Checks if the indicator can be normalized.

        Returns:
            (bool): True if the indicator can be normalized, False otherwise.
        """

        return True