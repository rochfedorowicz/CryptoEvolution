# tests/data_handling/test_coinbase_api_data_collector.py

# global imports
import asyncio
import logging
import pandas as pd
from ddt import data, ddt, unpack
from unittest import TestCase

# local imports
from source.data_handling import CoinbaseApiDataCollector
from source.utils import Granularity

@ddt
class CoinbaseApiDataCollectorTestCase(TestCase):
    """
    Test case for CoinbaseApiDataCollector class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: CoinbaseApiDataCollector = CoinbaseApiDataCollector()

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    def __update_sut(self, **kwargs) -> None:
        """
        Allows to update already created sut. It speeds up test
        cases' scenarios by enabling injections of certain values
        also into private sut members.
        """

        for name, value in kwargs.items():
            for attribute_name in self.__sut.__dict__:
                if name in attribute_name:
                    setattr(self.__sut, attribute_name, value)

    def test_coinbase_handler__collect_data_for_ticker(self) -> None:
        """
        Tests the _collect_data_for_ticker method of CoinbaseApiDataCollector.

        Verifies that the _collect_data_for_ticker method retrieves the correct candle data
        for a given trading pair and date range. The expected data is hardcoded for the
        date range from '2020-03-01' to '2020-03-03' for the 'BTC-USD' ticker with a granularity
        of one day.

        Asserts:
            The result DataFrame matches the expected DataFrame.
        """

        logging.info("Attempting to retrieve candle data for BTC-USD.")
        expected_data = pd.DataFrame(data = {
            'low': [8400.00, 8487.33, 8635.31],
            'high': [8752.34, 8973.45, 8927.45],
            'open': [8523.33, 8522.30, 8919.21],
            'close': [8522.31, 8915.00, 8757.84],
            'volume': [7353.139605, 10216.692545, 9152.706926]
        }, index = pd.DatetimeIndex(['2020-03-01', '2020-03-02', '2020-03-03'], name = 'time'))

        logging.info("Invoking _collect_data_for_ticker method.")
        result  = asyncio.run(self.__sut._collect_data_for_ticker('BTC-USD', '2020-03-01 00:00:00',
                                                                   '2020-03-03 00:00:00', Granularity.ONE_DAY))

        logging.info("Verifying the result DataFrame against expected values.")
        pd.testing.assert_frame_equal(result[0], expected_data)
        self.assertTrue('normalization_groups' in result[1])

    @data(
        ('BTC-USD', True),
        ('ETH-USD', True),
        ('INVALID-PAIR', False)
    )
    @unpack
    def test_coinbase_handler__validate_ticker(self, ticker: str, expected_result: bool) -> None:
        """
        Tests the _validate_ticker method of CoinbaseApiDataCollector.

        Verifies that the _validate_ticker method retrieves the correct trading pairs
        available on the exchange to validate given ticker.

        Asserts:
            The result DataFrame matches the expected DataFrame.
        """

        logging.info("Attempting to retrieve possible trading pairs.")

        logging.info("Invoking _validate_ticker method.")
        result = asyncio.run(self.__sut._validate_ticker(ticker))

        logging.info("Verifying the result DataFrame against expected values.")
        self.assertEqual(result, expected_result)
