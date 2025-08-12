# tests/plotting/test_reinforcement_training_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
from ddt import ddt
from unittest import TestCase

# local imports
from source.agent import ReinforcementLearningStrategyHandler
from source.plotting import ReinforcementTrainingPlotResponsibilityChain

@ddt
class ReinforcementTrainingPlotResponsibilityChainTestCase(TestCase):
    """
    Test case for ReinforcementTrainingPlotResponsibilityChain class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: ReinforcementTrainingPlotResponsibilityChain = ReinforcementTrainingPlotResponsibilityChain(3)

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")
        plt.close('all')

    def test_reinforcement_training_plot_responsibility_chain_plot__able_to_handle(self) -> None:
        """
        Tests ReinforcementTrainingPlotResponsibilityChain's plot functionality with valid input.

        Verifies that the plot method correctly handles input data with the 'reinforcement_learning'
        key and generates a plot with the expected title, labels, and data series. Tests that
        the plot contains two lines representing episode rewards and their moving average.

        Asserts:
            The plot has the correct title, axis labels, and number of data series.
            The data points in each plot match the input data.
        """

        logging.info("Starting plot test.")
        mocked_input_data = {
            'key': ReinforcementLearningStrategyHandler.PLOTTING_KEY,
            'plot_data': {
                'episode_reward': [2.0, 3.0, -2.0, 2.0, -1.0],
                'nb_steps': [4, 5, 6, 8, 9]
            }
        }

        expected_title = "Training history"
        expected_xlabel = "Trading points"
        expected_ylabel = "Reward"
        expected_number_of_plotted_lines = 2
        expected_xydata_first_line = [list(tup) for tup in list(zip([0] + mocked_input_data['plot_data']['nb_steps'],
                                                                    [0] + mocked_input_data['plot_data']['episode_reward']))]

        logging.info("Plotting using provided data.")
        result = self.__sut.plot(mocked_input_data)
        plotted_lines = result.get_lines()

        logging.info("Checking expected plot.")
        self.assertEqual(result.get_title(), expected_title)
        self.assertEqual(result.get_xlabel(), expected_xlabel)
        self.assertEqual(result.get_ylabel(), expected_ylabel)
        self.assertEqual(len(plotted_lines), expected_number_of_plotted_lines)
        self.assertEqual(plotted_lines[0].get_xydata().tolist(), expected_xydata_first_line)


    def test_reinforcement_training_plot_responsibility_chain_plot__unable_to_handle(self) -> None:
        """
        Tests ReinforcementTrainingPlotResponsibilityChain's plot functionality with invalid input.

        Verifies that the plot method correctly handles the case where the key does not
        match 'reinforcement_learning'. In this case, the handler should not process the request
        and should return None, indicating that the request should be passed to the next
        handler in the chain.

        Asserts:
            The method returns None when given an unrecognized key.
        """

        mocked_input_data = {
            'key': 'unknown_plot_type',
            'plot_data': None
        }

        logging.info("Plotting using provided data.")
        result = self.__sut.plot(mocked_input_data)

        logging.info("Checking if plot is empty.")
        self.assertEqual(result, None)