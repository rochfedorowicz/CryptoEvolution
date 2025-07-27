# indicators/relative_strength_index_indicator_handler.py

# global imports
import pandas as pd

# local imports
from source.indicators import IndicatorHandlerBase

class RelativeStrengthIndexIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Relative Strength Index (RSI) indicator. It measures the speed and change
    of price movements.
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
        Calculates Relative Strength Index (RSI) indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated RSI values.
        """

        rsi_data_df = pd.DataFrame(index = data.index)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window = self.__window_size, min_periods = 1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = self.__window_size, min_periods = 1).mean()
        rs = gain / loss
        rsi_data_df['rsi'] = 1 - (1 / (1 + rs))
        rsi_data_df.fillna(0.5, inplace = True)

        return rsi_data_df