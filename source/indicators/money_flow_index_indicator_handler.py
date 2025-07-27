# indicators/money_flow_index_indicator_handler.py

# global imports
import pandas as pd

# local imports
from source.indicators import IndicatorHandlerBase

class MoneyFlowIndexIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Money Flow Index (MFI) indicator. It measures the flow of money into and out of an asset.
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
        Calculates Money Flow Index (MFI) indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated MFI values.
        """

        typical_price = (data['high'] + data['low'] + data['close']) / 3
        delta = typical_price.diff()
        money_flow = typical_price * data['volume']
        positive_flow = money_flow.where(delta > 0, 0).rolling(window = self.__window_size, min_periods = 1).sum()
        negative_flow = money_flow.where(delta < 0, 0).rolling(window = self.__window_size, min_periods = 1).sum()

        mfi_data_df = pd.DataFrame(index = data.index)
        mfi_data_df['mfi'] = 1 - (1 / (1 + (positive_flow / negative_flow)))
        mfi_data_df.fillna(0.5, inplace = True)

        return mfi_data_df