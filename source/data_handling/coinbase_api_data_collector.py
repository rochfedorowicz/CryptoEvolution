# data_handling/coinbase_api_data_collector.py

# global imports
import aiohttp
import asyncio
import math
import pandas as pd
import pytz

# local imports
from source.data_handling import ApiDataCollectorBase
from source.utils import Granularity

class CoinbaseApiDataCollector(ApiDataCollectorBase):
    """
    Implements a data collector for Coinbase API.
    Responsible for collecting historical data for a given trading pair.
    """

    # local constants
    __MAX_NUMBER_OF_CANDLES_PER_REQUEST = 300
    __PRODUCTS_URL = 'https://api.exchange.coinbase.com/products'

    async def __send_request_to_coinbase(self, session: aiohttp.ClientSession, url: str, pid: int) -> list:
        """
        Sends request towards Coinbase API and handles exceedance of public rates by repeating
        request after certain time.

        Parameters:
            session (aiohttp.ClientSession): Session used to send request with.
            url (str): URL address that certain request is sent towards.
            pid (int): Request indentification number.

        Raises:
            RuntimeError: If public rates are exceeded. Will try to handle that and reattempt
                to sent request.

        Returns:
            (list): List of values returned by Coinbase API for certain request.
        """

        try:
            async with session.get(url) as response:
                 data = await response.json()
                 if 'message' in data and data['message'] == 'Public rate limit exceeded':
                     raise RuntimeError("Exceeded public rate! Retrying in 5s...")
                 return data
        except:
            await asyncio.sleep(5)
            return await self.__send_request_to_coinbase(session, url, pid)

    async def __get_possible_pairs(self) -> pd.DataFrame:
        """
        Collects data from Coinbase API regarding all possible trading pairs.

        Returns:
            (pd.DataFrame): Fetched possible trading pairs inside data frame.
        """

        async with aiohttp.ClientSession() as session:
            response = await asyncio.gather(self.__send_request_to_coinbase(session, self.__PRODUCTS_URL, 0))
            data = [[product['id'], product['base_currency'], product['quote_currency']] for product in response[0]]
            df = pd.DataFrame(sorted(data), columns = ['id', 'base_currency', 'quote_currency'])
            df.set_index('id', inplace = True)
            return df

    async def _validate_ticker(self, ticker: str) -> bool:
        """
        Validates if the ticker is supported by the API.

        Parameters:
            ticker (str): Stock ticker to validate.

        Returns:
            bool: True if ticker is valid, False otherwise.
        """

        df = await self.__get_possible_pairs()
        if ticker not in df.index:
            return False

        return True

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

        start_timestamp = int(self._convert_date_to_datetime(start_date).replace(tzinfo = pytz.UTC).timestamp())
        end_timestamp = int(self._convert_date_to_datetime(end_date).replace(tzinfo = pytz.UTC).timestamp())
        granularity_seconds = granularity.value
        total_periods = (end_timestamp - start_timestamp) // granularity_seconds
        requests_needed = math.ceil(total_periods / self.__MAX_NUMBER_OF_CANDLES_PER_REQUEST)

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(requests_needed):
                start_period = start_timestamp + i * self.__MAX_NUMBER_OF_CANDLES_PER_REQUEST * granularity_seconds
                end_period = min(start_period + self.__MAX_NUMBER_OF_CANDLES_PER_REQUEST * granularity_seconds, end_timestamp)
                url = f'{self.__PRODUCTS_URL}/{ticker}/candles?start={start_period}&end={end_period}&granularity={granularity_seconds}'
                tasks.append(self.__send_request_to_coinbase(session, url, i))

            responses = await asyncio.gather(*tasks)
            candles = [item for sublist in responses if sublist for item in sublist]
            df = pd.DataFrame(candles, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit = 's')
            df.set_index('time', inplace = True)
            df.sort_values(by = 'time', inplace = True)
            return df, { 'normalization_groups': [['low', 'high', 'open', 'close'], ['volume']] }
