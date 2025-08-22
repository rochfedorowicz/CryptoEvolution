# data_handling/data_handler.py

# global imports
import io
import logging
import os
import pandas as pd
from typing import Any, Optional

# local imports
from source.data_handling import ApiDataCollectorBase
from source.indicators import IndicatorHandlerBase
from source.utils import Granularity, SingletonMeta

class DataHandler(metaclass = SingletonMeta):
    """
    Responsible for data handling. Including data collection and preparation.
    """

    # local constants
    __EXPECTED_COLUMN_NAMES = ['time', 'low', 'high', 'open', 'close', 'volume']

    def __init__(self) -> None:
        """
        Class constructor. Initializes components needed for data handling.
        """

        self.__api_data_collectors: list[ApiDataCollectorBase] = []

    def register_api_data_collectors(self, api_data_collectors: list[ApiDataCollectorBase]) -> None:
        """
        Registers API data collectors for data collection.

        Parameters:
            api_data_collectors (list[ApiDataCollectorBase]): A list of instances of ApiDataCollectorBase or its subclasses.
        """

        for api_data_collector in api_data_collectors:
            if not isinstance(api_data_collector, ApiDataCollectorBase):
                raise TypeError("Parameter api_data_collector must be an instance of ApiDataCollectorBase or its subclass.")

        self.__api_data_collectors = api_data_collectors

    async def prepare_data(self, input_source: str, start_date: str, end_date: str, granularity: Optional[Granularity] = None,
        list_of_indicators: Optional[list[IndicatorHandlerBase]] = None) -> pd.DataFrame:
        """
        Collects data from coinbase API and extends it with list of indicators.

        Parameters:
            input_source (str): String representing unique trading symbol or path
                to the file to be preprocessed.
            start_date (str): String representing date that collected data should start from.
            end_date (str): String representing date that collected data should finish at.
            granularity (Optional[Granularity]): Enum specifying resolution of collected data - e.g. each
                15 minutes or 1 hour or 6 hours is treated separately. It is optional when input_source is a file path,
                but must be provided when input_source is a ticker.
            list_of_indicators (Optional[list[IndicatorHandlerBase]]): List of indicators that should be
                calculated and added to the data. Defaults to None, which means no indicators
                will be added.

        Raises:
            RuntimeError: If given trading pair symbol is not recognized.

        Returns:
            (pd.DataFrame): Preprocessed data extended with given indicators.
        """

        data, meta_data = None, None
        if list_of_indicators is None:
            list_of_indicators = []

        # Input source is a file path
        if os.path.isfile(input_source):
            logging.info(f"Assuming input source '{input_source}' to be a file path.")
            if input_source.endswith('.csv'):
                data = pd.read_csv(input_source)
                data.columns = data.columns.str.lower()

                if not all(col in data.columns for col in self.__EXPECTED_COLUMN_NAMES):
                    logging.error(f"Found columns: {data.columns.tolist()}, "
                                  f"while expected columns are: {self.__EXPECTED_COLUMN_NAMES}")
                    raise ValueError(f"CSV file must contain columns: {', '.join(self.__EXPECTED_COLUMN_NAMES)}")

                data = data[self.__EXPECTED_COLUMN_NAMES]
                data['time'] = pd.to_datetime(data['time'])
                data.set_index('time', inplace = True)
                data.sort_index(inplace = True)

                data = data[(data.index >= pd.to_datetime(start_date)) & \
                            (data.index < pd.to_datetime(end_date))]
                meta_data = { 'normalization_groups': [['low', 'high', 'open', 'close'], ['volume']] }
            else:
                raise ValueError("Unsupported file format. Please provide a CSV file.")

        # Input source is assumed to be a ticker otherwise
        else:
            logging.info(f"Assuming input source '{input_source}' to be a ticker.")
            if granularity is None:
                raise ValueError("Granularity must be provided when input source is a ticker.")

            for api_data_collector in self.__api_data_collectors:
                try:
                    data, meta_data = await api_data_collector.collect_data(input_source, start_date, end_date, granularity)
                    break
                except Exception:
                    logging.info(f"Did not manage to collect data for {input_source} using "
                                    f"{api_data_collector.__class__.__name__}... trying next one.")

            if data is None or meta_data is None:
                raise RuntimeError('Trading pair not recognized!')

            if data.empty:
                raise RuntimeError(f'No data collected for {input_source} between {start_date} and {end_date} with granularity {granularity}.')

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
