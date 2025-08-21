# indicators/on_balance_volume_indicator_handler.py

# global imports
import pandas as pd

# local imports
from source.indicators import IndicatorHandlerBase

class OnBalanceVolumeIndicatorHandler(IndicatorHandlerBase):
    """
    Implements On-Balance Volume (OBV) indicator. It uses volume flow to predict changes in stock price.
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates On-Balance Volume (OBV) indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated On-Balance Volume values.
        """

        coeff = data['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        obv_data_df = pd.DataFrame(index = data.index)
        obv_data_df['obv'] = (data['volume'] * coeff).cumsum()

        return obv_data_df

    def can_be_normalized(self) -> bool:
        """
        Checks if the indicator can be normalized.

        Returns:
            (bool): True if the indicator can be normalized, False otherwise.
        """

        return True