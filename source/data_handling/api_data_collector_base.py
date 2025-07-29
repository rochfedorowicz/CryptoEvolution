# data_handling/api_data_collector_base.py

# global imports
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime

# local imports
from source.utils import Granularity

class ApiDataCollectorBase(ABC):
    """
    Implements a base class for collecting data from various APIs.
    """

    @abstractmethod
    async def _validate_ticker(self, ticker: str) -> bool:
        """
        Validates if the ticker is supported by the API.

        Parameters:
            ticker (str): Stock ticker to validate.

        Returns:
            bool: True if ticker is valid, False otherwise.
        """

        pass

    @abstractmethod
    async def _collect_data_for_ticker(self, ticker: str, start_date: str, end_date: str,
        granularity: Granularity) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Collects data for a specific ticker from the API.

        Parameters:
            ticker (str): Stock ticker to collect data for.
            start_date (str): Start date for the data collection.
            end_date (str): End date for the data collection.
            granularity (Granularity): Data resolution.

        Returns:
            tuple[pd.DataFrame, dict[str, str]]: A tuple containing the collected data and metadata.
        """

        pass

    def _convert_date_to_datetime(self, date_str: str, date_format: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """
        Converts date string to datetime object.

        Parameters:
            date_str (str): String representing date.
            date_format (str): Format of the date string.

        Returns:
            datetime: Converted datetime object.
        """
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            raise ValueError(f'Invalid date format! Expected was {date_format}.')

    async def collect_data(self, ticker: str, start_date: str, end_date: str,
        granularity: Granularity) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Collects data for a specific ticker from the API.

        Parameters:
            ticker (str): Stock ticker to collect data for.
            start_date (str): Start date for the data collection.
            end_date (str): End date for the data collection.
            granularity (Granularity): Data resolution.

        Raises:
            ValueError: If the ticker is not supported by the API or if the granularity is not valid.

        Returns:
            tuple[pd.DataFrame, dict[str, str]]: A tuple containing the collected data and metadata.
        """

        if granularity not in Granularity:
            raise ValueError(f"{granularity} is not a valid value of Granularity enum!")

        if not await self._validate_ticker(ticker):
            raise ValueError(f"Ticker {ticker} is not supported by this API.")

        return await self._collect_data_for_ticker(ticker, start_date, end_date, granularity)
