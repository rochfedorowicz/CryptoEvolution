# tests/plotting/test_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
from ddt import data, ddt, unpack
from unittest import TestCase
from unittest.mock import Mock

# local imports
from source.plotting import PlotResponsibilityChainBase

class TestPlotResponsibilityChain(PlotResponsibilityChainBase):
    def _can_plot(self, _):
        pass

    def _plot(self, _):
        pass

@ddt
class PlotResponsibilityChainTestCase(TestCase):
    """
    Test case for PlotResponsibilityChain pattern implementation. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__chain_list: list[PlotResponsibilityChainBase] = []
        self.__chain_list.append(TestPlotResponsibilityChain())
        self.__chain_list.append(TestPlotResponsibilityChain())
        self.__chain_list.append(TestPlotResponsibilityChain())
        self.__chain_list[0].add_next_chain_link(self.__chain_list[1])
        self.__chain_list[0].add_next_chain_link(self.__chain_list[2])

        self.__chain_list[0]._can_plot = Mock(side_effect = lambda key: key == 'key_1')
        self.__chain_list[0]._plot = Mock(return_value = plt.subplots()[1])
        self.__chain_list[1]._can_plot = Mock(side_effect = lambda key: key == 'key_2')
        self.__chain_list[1]._plot = Mock(return_value = plt.subplots()[1])
        self.__chain_list[2]._can_plot = Mock(side_effect = lambda key: key == 'key_3')
        self.__chain_list[2]._plot = Mock(return_value = plt.subplots()[1])

        self.__sut: PlotResponsibilityChainBase = self.__chain_list[0]

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")
        plt.close('all')

    @data(
        ('key_1', plt.Axes, 0),
        ('key_2', plt.Axes, 1),
        ('key_3', plt.Axes, 2),
        ('key_4', type(None), -1)  # Unrecognized key
    )
    @unpack
    def test_plot_responsibility_chain_plot(self, key: str, expected_result_type: type, index: int) -> None:
        """
        Tests the responsibility chain pattern implementation for plot handling.

        Verifies that the plot method correctly handles requests by finding the
        appropriate handler in the chain based on the provided key. Tests that
        requests with matching keys are processed by the corresponding handler
        and that unrecognized keys result in None being returned.

        Parameters:
            key (str): The key to test with the responsibility chain.
            expected_result_type (type): Expected return type based on the key.

        Asserts:
            The result is of the expected type (plt.Axes for recognized keys,
            None for unrecognized keys).
        """

        logging.info("Starting plot test.")
        mocked_input_data = {
            'key': key,
            'plot_data': []
        }

        logging.info("Plotting using provided data.")
        result = self.__sut.plot(mocked_input_data)

        logging.info("Checking expected result type.")
        self.assertTrue(isinstance(result, expected_result_type))
        if index >= 0:
            for i in range(index + 1):
                self.__chain_list[i]._can_plot.assert_called_once_with(key)
            self.__chain_list[index]._plot.assert_called_once()