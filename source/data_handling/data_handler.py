# data_handling/data_handler.py

# global imports
import pandas as pd
from typing import Optional

# local imports
from source.data_handling import CoinBaseHandler
from source.indicators import IndicatorHandlerBase
from source.utils import Granularity

class DataHandler():
    """
    Responsible for data handling. Including data collection and preparation.
    """

    def __init__(self, list_of_indicators_to_apply: Optional[list[IndicatorHandlerBase]] = None) -> None:
        """
        Class constructor.

        Parameters:
            list_of_indicators_to_apply (list): List of indicators further to apply.
        """

        if list_of_indicators_to_apply is None:
            list_of_indicators_to_apply = []
        self.__indicators: list[IndicatorHandlerBase] = list_of_indicators_to_apply
        self.__coinbase: CoinBaseHandler = CoinBaseHandler()

    async def prepare_data(self, trading_pair: str, start_date: str, end_date: str, granularity: Granularity) -> pd.DataFrame:
        """
        Collects data from coinbase API and extends it with assigned list of indicators.

        Parameters:
            trading_pair (str): String representing unique trainding pair symbol.
            start_date (str): String representing date that collected data should start from.
            end_date (str): String representing date that collected data should finish at.
            granularity (Granularity): Enum specifying resolution of collected data - e.g. each
                15 minutes or 1 hour or 6 hours is treated separately

        Raises:
            RuntimeError: If given traiding pair symbol is not recognized.

        Returns:
            (pd.DataFrame): Collected data extended with given indicators.
        """

        possible_traiding_pairs = await self.__coinbase.get_possible_pairs()
        if trading_pair not in possible_traiding_pairs.index:
            raise RuntimeError('Traiding pair not recognized!')

        data = await self.__coinbase.get_candles_for(trading_pair, start_date, end_date, granularity)
        if len(self.__indicators) > 0:
            indicators_data = []
            for indicator in self.__indicators:
                indicators_data.append(indicator.calculate(data))
            data = pd.concat([data] + indicators_data, axis=1)

        return data
