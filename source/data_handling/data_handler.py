# data_handling/data_handler.py

# global imports
import io
import pandas as pd
from typing import Any, Optional

# local imports
from source.data_handling import CoinBaseHandler
from source.indicators import IndicatorHandlerBase
from source.utils import Granularity, SingletonMeta

class DataHandler(metaclass = SingletonMeta):
    """
    Responsible for data handling. Including data collection and preparation.
    """

    def __init__(self) -> None:
        """
        Class constructor. Initializes needed components for data handling.
        """

        self.__coinbase: CoinBaseHandler = CoinBaseHandler()

    async def prepare_data(self, trading_pair: str, start_date: str, end_date: str,
        granularity: Granularity, list_of_indicators: Optional[list[IndicatorHandlerBase]] = None) \
        -> pd.DataFrame:
        """
        Collects data from coinbase API and extends it with list of indicators.

        Parameters:
            trading_pair (str): String representing unique trading pair symbol.
            start_date (str): String representing date that collected data should start from.
            end_date (str): String representing date that collected data should finish at.
            granularity (Granularity): Enum specifying resolution of collected data - e.g. each
                15 minutes or 1 hour or 6 hours is treated separately
            list_of_indicators (list[IndicatorHandlerBase]): List of indicators that should be
                calculated and added to the data. Defaults to None, which means no indicators
                will be added.

        Raises:
            RuntimeError: If given trading pair symbol is not recognized.

        Returns:
            (pd.DataFrame): Collected data extended with given indicators.
        """

        if list_of_indicators is None:
            list_of_indicators = []

        possible_trading_pairs = await self.__coinbase.get_possible_pairs()
        if trading_pair not in possible_trading_pairs.index:
            raise RuntimeError('Trading pair not recognized!')

        data, meta_data = await self.__coinbase.get_candles_for(trading_pair, start_date, end_date, granularity)
        if len(list_of_indicators) > 0:
            indicators_data = []
            for indicator in list_of_indicators:
                indicators_data.append(indicator.calculate(data))
                if indicator.can_be_normalized():
                    columns = indicators_data[-1].columns.tolist()
                    meta_data['normalization_groups'].append(columns)
            data = pd.concat([data] + indicators_data, axis = 1)

        return data, meta_data

    def save_extended_data_into_csv_formatted_string_buffer(self, data: pd.DataFrame,
        meta_data: Optional[dict[str, Any]] = None) -> io.StringIO:
        """
        Saves extended data into a CSV formatted string buffer.

        Parameters:
            data (pd.DataFrame): Data frame to be saved.
            meta_data (Optional[dict[str, Any]]): Optional metadata to include in the CSV.

        Returns:
            (io.StringIO): StringIO buffer containing the CSV formatted data.
        """

        file_content_string_buffer = io.StringIO()

        if meta_data is not None:
            file_content_string_buffer.write(f'# {meta_data} \n')

        data.to_csv(file_content_string_buffer, index = True)

        return file_content_string_buffer

    def read_extended_data_from_csv_formatted_string_buffer(self,
        file_content_string_buffer: io.StringIO) -> tuple[pd.DataFrame, Optional[dict[str, Any]]]:
        """
        Reads extended data from a CSV formatted string buffer.

        Parameters:
            file_content_string_buffer (io.StringIO): StringIO buffer containing the CSV formatted data.

        Returns:
            (tuple[pd.DataFrame, Optional[dict[str, Any]]]): Tuple containing the data frame with extended data
            and optional metadata.
        """

        meta_data = None

        first_line = file_content_string_buffer.readline().strip()
        if first_line.startswith('#'):
            meta_data = eval(first_line[1:])

        data = pd.read_csv(file_content_string_buffer)

        return data, meta_data
