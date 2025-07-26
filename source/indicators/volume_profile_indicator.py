# indicators/volume_profile_indicator.py

# global imports
import numpy as np
import pandas as pd
from collections import defaultdict

# local imports
from source.indicators import IndicatorHandlerBase

class VolumeProfileIndicatorHandler(IndicatorHandlerBase):
    """
    Implements static volume profile indicator. It denotes volume traded at
    certain price levels. Calculated data can not be directly mapped to input
    data and should be treated as the separate chart.
    """

    def __init__(self, number_of_steps: int = 40) -> None:
        """
        Class constructor.

        Parameters:
            number_of_steps (int): Number of bins that price should be put into
                while creating volume profile.
        """

        self.number_of_steps = number_of_steps

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates static volume profile indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated static volume profile values.
        """

        volume_profile = defaultdict(float)
        data_min = data['low'].min()
        data_max = data['high'].max()
        step = (data_max - data_min) / (self.number_of_steps - 1)

        for _, row in data.iterrows():
            equalized_low = row['low'] // step * step
            equalized_high = row['high'] // step * step
            price_range = np.linspace(equalized_low, equalized_high, num = int(round((equalized_high - equalized_low) / step + 1, 0)))
            price_range = np.round(price_range, 6 - int(np.floor(np.log10(data_min))))
            volume_per_step = row['volume'] / len(price_range)

            for price in price_range:
                volume_profile[price] += volume_per_step

        profile_df = pd.DataFrame(list(volume_profile.items()), columns = ['price', 'volume'])
        profile_df.sort_values(by = 'price', inplace=True)

        return profile_df