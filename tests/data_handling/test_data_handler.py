# tests/data_handling/test_data_handler.py

# global imports
import asyncio
import io
import logging
import pandas as pd
from ddt import ddt
from unittest import TestCase
from unittest.mock import Mock, AsyncMock, patch

# local imports
from source.data_handling import DataHandler
from source.indicators import IndicatorHandlerBase
from source.utils import Granularity

class TestIndicatorHandler(IndicatorHandlerBase):
    def calculate(self, _):
        pass

@ddt
class DataHandlerTestCase(TestCase):
    """
    Test case for DataHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    # local constants
    __MOCKED_COINBASE_HANDLER_DATA = pd.DataFrame(data = {
        'low': [8400.00, 8487.33, 8635.31],
        'high': [8752.34, 8973.45, 8927.45],
        'open': [8523.33, 8522.30, 8919.21],
        'close': [8522.31, 8915.00, 8757.84],
        'volume': [7353.139605, 10216.692545, 9152.706926]
    }, index = pd.DatetimeIndex(['2020-03-01', '2020-03-02', '2020-03-03'], name = 'time'))
    __MOCKED_COINBASE_HANDLER_META_DATA = {
        'normalization_groups': [ ['some_column_1', 'some_column_2'] ]
    }

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: DataHandler = DataHandler()

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

    @patch('source.data_handling.CoinBaseHandler.get_candles_for', new_callable = AsyncMock)
    def test_data_handler_prepare_data__no_indicators(self, mocked_get_candles_for) -> None:
        """
        Tests the prepare_data method of DataHandler without indicators.

        Verifies that the prepare_data method do not modify data when there is no specified
        indicators to the data. The get_candles_for method of CoinBaseHandler is mocked to
        return predefined data.

        Expected Result:
            Same as mocked data.

        Asserts:
            The result DataFrame matches the expected DataFrame.
            The get_candles_for method was called once.
        """

        logging.info("Attempting to retrieve data for BTC-USD.")
        mocked_get_candles_for.return_value = (self.__MOCKED_COINBASE_HANDLER_DATA,
                                               self.__MOCKED_COINBASE_HANDLER_META_DATA)
        expected_data = self.__MOCKED_COINBASE_HANDLER_DATA

        logging.info("Invoking prepare_data method without indicators.")
        result = asyncio.run(self.__sut.prepare_data('BTC-USD', '2020-03-01 00:00:00', '2020-03-03 00:00:00',
                                                     Granularity.ONE_DAY))

        logging.info("Validating the results.")
        pd.testing.assert_frame_equal(result[0], expected_data)
        self.assertTrue('normalization_groups' in result[1])
        mocked_get_candles_for.assert_called_once()

    @patch('source.data_handling.CoinBaseHandler.get_candles_for', new_callable = AsyncMock)
    def test_data_handler_prepare_data__with_indicators(self, mocked_get_candles_for) -> None:
        """
        Tests the prepare_data method of DataHandler with indicators.

        Verifies that the prepare_data method applies the specified indicators to the data.
        The get_candles_for method of CoinBaseHandler is mocked to return predefined data,
        and two indicators are applied.

        Expected Result:
            Original data combined with the results of the mean_high and std_low indicators.

        Asserts:
            The result DataFrame is extended.
            The get_candles_for method was called once.
        """

        logging.info("Attempting to retrieve data for BTC-USD with indicators.")
        mocked_get_candles_for.return_value = (self.__MOCKED_COINBASE_HANDLER_DATA,
                                               self.__MOCKED_COINBASE_HANDLER_META_DATA)
        mean_high_mock_indicator = Mock(spec = TestIndicatorHandler)
        mean_high_mock_indicator.calculate = lambda data: \
            data['high'].rolling(window = 2).mean().to_frame(name = 'mean_high')
        std_low_mock_indicator = Mock(spec = TestIndicatorHandler)
        std_low_mock_indicator.calculate = lambda data: \
            data['low'].rolling(window = 2).std().to_frame(name = 'std_low')

        expected_data = pd.concat([self.__MOCKED_COINBASE_HANDLER_DATA,
                                  mean_high_mock_indicator.calculate(self.__MOCKED_COINBASE_HANDLER_DATA),
                                  std_low_mock_indicator.calculate(self.__MOCKED_COINBASE_HANDLER_DATA)], axis = 1)
        indicators = [mean_high_mock_indicator, std_low_mock_indicator]

        logging.info("Invoking prepare_data method with indicators.")
        result = asyncio.run(self.__sut.prepare_data('BTC-USD', '2020-03-01 00:00:00', '2020-03-03 00:00:00',
                                                     Granularity.ONE_DAY, indicators))

        logging.info("Validating the results with indicators.")
        pd.testing.assert_frame_equal(result[0], expected_data)
        self.assertTrue('normalization_groups' in result[1])
        mocked_get_candles_for.assert_called_once()

    def test_data_handler_save_extended_data_into_csv_formatted_string_buffer(self) -> None:
        """
        Tests the save_extended_data_into_csv_formatted_string_buffer method of DataHandler.

        Verifies that the method correctly saves the extended data and metadata into a CSV formatted string buffer.

        Asserts:
            The CSV formatted string buffer matches the expected format.
        """

        logging.info("Attempting to save data into CSV formatted string buffer.")
        expected_result = f"# {self.__MOCKED_COINBASE_HANDLER_META_DATA} \n" + \
                          self.__MOCKED_COINBASE_HANDLER_DATA.to_csv(index = True, header = True)

        logging.info("Invoking save_extended_data_into_csv_formatted_string_buffer.")
        result = self.__sut.save_extended_data_into_csv_formatted_string_buffer(self.__MOCKED_COINBASE_HANDLER_DATA,
                                                                                self.__MOCKED_COINBASE_HANDLER_META_DATA)

        logging.info("Validating the CSV formatted string buffer.")
        self.assertEqual(result.getvalue(), expected_result)

    def test_data_handler_read_extended_data_from_csv_formatted_string_buffer(self) -> None:
        """"""

        logging.info("Attempting to read data from CSV formatted string buffer.")
        mocked_input = io.StringIO(f"# {self.__MOCKED_COINBASE_HANDLER_META_DATA} \n" +
                                  self.__MOCKED_COINBASE_HANDLER_DATA.to_csv(index = True, header = True))

        logging.info("Invoking read_extended_data_from_csv_formatted_string_buffer.")
        result = self.__sut.read_extended_data_from_csv_formatted_string_buffer(mocked_input)
        result[0]['time'] = pd.to_datetime(result[0]['time'])
        result[0].set_index('time', inplace = True)

        logging.info("Validating the read data and metadata.")
        pd.testing.assert_frame_equal(result[0], self.__MOCKED_COINBASE_HANDLER_DATA)
        self.assertEqual(result[1], self.__MOCKED_COINBASE_HANDLER_META_DATA)