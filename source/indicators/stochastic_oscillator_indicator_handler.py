# indicators/stochastic_oscillator_indicator_handler.py

# global imports
import pandas as pd

# local imports
from source.indicators import IndicatorHandlerBase

class StochasticOscillatorIndicatorHandler(IndicatorHandlerBase):
    """
    Implements stochastic oscillator indicator. It describes how expensive or cheap
    current price is over the given period. Result of calculation is percentage
    assuming values over 80% for expensive and under 20% for cheap price.
    """

    def __init__(self, window_size: int = 14, d_period: int = 3) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Length of window that indicator should be applied over.
            d_period (int): Length of smoothing window that should be applied over indicator data.
        """

        self.__window_size = window_size
        self.__d_period = d_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates stochastic oscillator indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated stochastic oscillator values.
        """

        high_series = data['high']
        low_series = data['low']
        close_series = data['close']

        highest_high = high_series.rolling(window = self.__window_size, min_periods = 1).max()
        lowest_low = low_series.rolling(window = self.__window_size, min_periods = 1).min()

        stochastic_data_df = pd.DataFrame(index = data.index)
        stochastic_data_df['so_k'] = (close_series - lowest_low) / (highest_high - lowest_low)
        stochastic_data_df['so_d'] = stochastic_data_df['so_k']. \
            rolling(window = self.__d_period, min_periods = 1).mean()

        return stochastic_data_df