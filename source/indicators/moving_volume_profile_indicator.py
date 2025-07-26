# indicators/moving_volume_profile_indicator.py

# global imports
import pandas as pd

# local imports
from source.indicators import VolumeProfileIndicatorHandler

class MovingVolumeProfileIndicatorHandler(VolumeProfileIndicatorHandler):
    """
    Implements moving volume profile indicator. Uses normal volume profile indicator
    to calculate volume profile for the given window size. It denotes volume traded
    at certain price over the period given by window size.
    """

    def __init__(self, window_size: int = 14, number_of_steps: int = 40) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Length of window that indicator should be applied over.
            number_of_steps (int): Number of bins that price should be put into
                while creating volume profile for certain window.
        """

        super().__init__(number_of_steps)
        self.window_size = window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates moving volume profile indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated moving volume profile values.
        """

        moving_volume_price_df = pd.DataFrame(index = data.index)

        for i, (index, row) in enumerate(data.iterrows()):
            volume_profiles_data = super().calculate(data[max(0, i - self.window_size) : i + 1])
            current_average_price = (row['low'] + row['high']) / 2
            volume_profiles_connected_to_lower_prices = volume_profiles_data['price'] <= current_average_price
            moving_volume_price_df.loc[index, 'moving_volume_profile'] = volume_profiles_data[volume_profiles_connected_to_lower_prices]['volume'].iloc[-1]

        return moving_volume_price_df

    def can_be_normalized(self) -> bool:
        """
        Checks if the indicator can be normalized.

        Returns:
            (bool): True if the indicator can be normalized, False otherwise.
        """

        return True