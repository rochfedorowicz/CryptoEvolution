# indicators/bollinger_bands_oscillator_indicator_handler.py

# global imports
import pandas as pd

# local imports
from source.indicators import IndicatorHandlerBase

class BollingerBandsDeviatorIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Bollinger Bands deviator indicator. It describes how far the current price is
    from the simple moving average (SMA) in terms of standard deviations.
    The Bollinger Bands itself is a volatility indicator that consists of two outer bands
    (standard deviations above and below the SMA) that Bollinger Bands deviator implementation
    uses for upper and lower boundaries.
    """

    def __init__(self, window_size: int = 20, std_coeff: int = 2) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Length of window that indicator should be applied over.
            std_coeff (int): Coefficient for standard deviation to define the width of the bands.
        """

        self.__window_size = window_size
        self.__std_coeff = std_coeff

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Bollinger Bands deviator indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated Bollinger Bands deviator values.
        """

        rolling_mean = data['close'].rolling(window = self.__window_size, min_periods = 1).mean()
        rolling_std = data['close'].rolling(window = self.__window_size, min_periods = 1).std()
        upper_band = rolling_mean + (self.__std_coeff * rolling_std)
        lower_band = rolling_mean - (self.__std_coeff * rolling_std)

        bollinger_data_df = pd.DataFrame(index = data.index)
        bollinger_data_df['bb_dev'] = (data['close'] - rolling_mean) / (upper_band - lower_band)
        bollinger_data_df.fillna(0, inplace = True)

        return bollinger_data_df