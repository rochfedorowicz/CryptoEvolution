# data_handling/yahoo_finance_api_data_collector.py

# global imports
import asyncio
import logging
import math
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# local imports
from source.data_handling.api_data_collector_base import ApiDataCollectorBase
from source.utils import Granularity

class YahooFinanceApiDataCollector(ApiDataCollectorBase):
    """
    Implements a data collector for Yahoo Finance API.
    Responsible for collecting historical data for a given ticker.
    """

    # local constants
    __MAX_NUMBER_OF_PAST_DAYS_PER_GRANULARITY = {
        Granularity.ONE_MINUTE: 8,       # 60 x 24 x 8 = 11520 data points
        Granularity.FIVE_MINUTES: 60,    # 12 x 24 x 60 = 17280 data points
        Granularity.FIFTEEN_MINUTES: 60, # 4 x 24 x 60 = 5760 data points
        Granularity.THIRTY_MINUTES: 60,  # 2 x 24 x 60 = 2880 data points
        Granularity.ONE_HOUR: 730,       # 1 x 24 x 730 = 17520 data points
        Granularity.SIX_HOURS: 730,      # 1 / 4 x 24 x 730 = 4380 data points
        Granularity.ONE_DAY: math.inf    # No limit for daily data
    }

    async def _validate_ticker(self, ticker: str) -> bool:
        """
        Validates if the ticker is supported by the API.

        Parameters:
            ticker (str): Stock ticker to validate.

        Returns:
            bool: True if ticker is valid, False otherwise.
        """

        try:
            ticker = yf.Ticker(ticker)
            info = ticker.info
            return 'symbol' in info or 'shortName' in info
        except:
            return False

    async def _collect_data_for_ticker(self, ticker: str, start_date: str, end_date: str, granularity: Granularity) \
        -> tuple[pd.DataFrame, dict[str, str]]:
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

        start_date = self._convert_date_to_datetime(start_date)
        end_date = self._convert_date_to_datetime(end_date)

        max_days = self.__MAX_NUMBER_OF_PAST_DAYS_PER_GRANULARITY[granularity]
        if math.isfinite(max_days) and datetime.now() - start_date > timedelta(days = max_days):
            logging.warning(f"Start date {start_date} is too recent for {ticker} with granularity {granularity}. "
                            f"Max allowed days is {max_days}, while requested was {(datetime.now() - start_date).days}.")
            return pd.DataFrame(), {}

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).history(
                start = start_date,
                end = end_date,
                interval = str(granularity) if granularity != Granularity.SIX_HOURS else '1h',
            )
        )

        if df.empty:
            logging.warning(f"Empty data returned for {ticker} between {start_date} and {end_date}.")
            return df, {}

        df = df.iloc[:, :-2].copy()
        df.columns = df.columns.str.lower()
        df.index.name = 'time'
        df.index = df.index.tz_convert('UTC').tz_localize(None)
        df.sort_values(by = 'time', inplace = True)
        metadata = {'normalization_groups': [['open', 'high', 'low', 'close'], ['volume']]}

        if granularity == Granularity.SIX_HOURS:
            df = df.resample('6H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            df.dropna(inplace = True)

        return df, metadata
