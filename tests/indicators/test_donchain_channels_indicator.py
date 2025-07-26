# tests/indicators/test_donchain_channels_indicator.py

# global imports
import logging
import pandas as pd
from ddt import ddt
from unittest import TestCase

# local imports
from source.indicators import DonchainChannelsIndicatorHandler

@ddt
class DonchainChannelsIndicatorHandlerTestCase(TestCase):
    """
    Test case for DonchainChannelsIndicatorHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    # local constants
    __INPUT_DATA = pd.DataFrame(data = {
        'low': [20000.0, 20500.0, 20100.0, 20100.0, 20000.0],
        'high': [20900.0, 20900.0, 21000.0, 20900.0, 21700.0],
        'open': [20050.0, 20600.0, 20400.0, 20800.0, 20200.0],
        'close': [20600.0, 20400.0, 20800.0, 20200.0, 20900.0],
        'volume': [1000.0, 1200.0, 1100.0, 1300.0, 900.0]
    })

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: DonchainChannelsIndicatorHandler = DonchainChannelsIndicatorHandler(3)

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

    def test_donchain_channels_indicator_calculate(self) -> None:
        """
        Tests the DonchainChannelsIndicatorHandler.

        Verifies that the DonchainChannelsIndicatorHandler calculates the upper,
        lower, and middle Donchian channels for given input data with a specified window size.

        Asserts:
            The result DataFrame matches the expected DataFrame.
        """

        logging.info("Attempting to calculate Donchain channels indicator.")
        expected = pd.DataFrame(data = {
            'upper_channel': [20900.0, 20900.0, 21000.0, 21000.0, 21700.0],
            'lower_channel': [20000.0, 20000.0, 20000.0, 20100.0, 20000.0],
            'middle_channel': [20450.0, 20450.0, 20500.0, 20550.0, 20850.0]
        })

        logging.info("Invoking calculate method.")
        result = self.__sut.calculate(data = self.__INPUT_DATA)

        logging.info("Validating expected results.")
        pd.testing.assert_frame_equal(result, expected)
