# indicators/donchain_channels_indicator.py

# global imports
import pandas as pd

# local imports
from source.indicators import IndicatorHandlerBase

class DonchainChannelsIndicatorHandler(IndicatorHandlerBase):
    """
    Implements donchain channels indicator. It indicates the highest and the
    lowest price for the certain period, creating so called channel. It also
    calculates average price as the average of the lowest and the highest price.
    """

    def __init__(self, window_size: int = 20) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Length of window that indicator should be applied over.
        """

        self.window_size = window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates donchain channels indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated donchain channels values.
        """

        donchian_df = pd.DataFrame(index = data.index)
        donchian_df['upper_channel'] = data['high'].rolling(window = self.window_size, min_periods = 1).max()
        donchian_df['lower_channel'] = data['low'].rolling(window = self.window_size, min_periods = 1).min()
        donchian_df['middle_channel'] = (donchian_df['upper_channel'] + donchian_df['lower_channel']) / 2

        return donchian_df

    def can_be_normalized(self) -> bool:
        """
        Checks if the indicator can be normalized.

        Returns:
            (bool): True if the indicator can be normalized, False otherwise.
        """

        return True