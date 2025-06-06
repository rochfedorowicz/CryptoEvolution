# tests/plotting/test_classification_training_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
import numpy as np
from ddt import ddt
from unittest import TestCase

# local imports
from source.agent import ClassificationLearningStrategyHandler
from source.model import SklearnModelAdapter, TFModelAdapter
from source.plotting import ClassificationTrainingPlotResponsibilityChain

@ddt
class ClassificationTrainingPlotResponsibilityChainTestCase(TestCase):
    """
    Test case for ClassificationTrainingPlotResponsibilityChain class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: ClassificationTrainingPlotResponsibilityChain = ClassificationTrainingPlotResponsibilityChain()

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")
        plt.close('all')

    def test_classification_training_plot_responsibility_chain_plot__tf_able_to_handle(self) -> None:
        """
        Tests ClassificationTrainingPlotResponsibilityChain's plot functionality with valid input.

        Verifies that the plot method correctly handles input data with the 'classification_learning_tensorflow'
        key and generates a plot with the expected title, labels, and data series. Tests that
        the plot contains three lines representing currency prices, asset values, and the
        moving average.

        Asserts:
            The plot has the correct title, axis labels, and number of data series.
            The data points in each plot match the input data.
        """

        mocked_input_data = {
            'key': ClassificationLearningStrategyHandler.PLOTTING_KEYS[1] + "_" + TFModelAdapter.TAG,
            'plot_data': {
                'loss': [0.1, 0.08, 0.05, 0.03],
                'accuracy': [0.9, 0.92, 0.95, 0.97],
                'val_loss': [0.12, 0.1, 0.07, 0.05],
                'val_accuracy': [0.88, 0.9, 0.93, 0.95]
            }
        }

        # Plot 1: Training loss and accuracy
        expected_training_metrics_ax_title = 'Training loss and accuracy'
        expected_training_metrics_ax_xlabel = 'Epoch'
        expected_training_metrics_ax_ylabel = 'Value'
        expected_training_metrics_ax_number_of_lines = 2
        loss = mocked_input_data['plot_data']['loss']
        accuracy = mocked_input_data['plot_data']['accuracy']
        expected_training_metrics_ax_xydata_first_line = [list(tup) for tup in
            list(zip(range(len(loss)), loss))]
        expected_training_metrics_ax_xydata_second_line = [list(tup) for tup in
            list(zip(range(len(accuracy)), accuracy))]

        # Plot 2: Validation loss and accuracy
        expected_validation_metrics_ax_title = 'Validation loss and accuracy'
        expected_validation_metrics_ax_xlabel = 'Epoch'
        expected_validation_metrics_ax_ylabel = 'Value'
        expected_validation_metrics_ax_number_of_lines = 2
        val_loss = mocked_input_data['plot_data']['val_loss']
        val_accuracy = mocked_input_data['plot_data']['val_accuracy']
        expected_validation_metrics_ax_xydata_first_line = [list(tup) for tup in
            list(zip(range(len(val_loss)), val_loss))]
        expected_validation_metrics_ax_xydata_second_line = [list(tup) for tup in
            list(zip(range(len(val_accuracy)), val_accuracy))]

        logging.info("Plotting using provided data.")
        _ = self.__sut.plot(mocked_input_data)
        fig = plt.gcf()

        # Plot 1: Training loss and accuracy
        training_metrics_ax = fig.axes[0]
        training_metrics_lines = training_metrics_ax.get_lines()

        # Plot 2: Validation loss and accuracy
        validation_metrics_ax = fig.axes[1]
        validation_metrics_lines = validation_metrics_ax.get_lines()

        logging.info("Checking expected plots.")
        self.assertEqual(len(fig.axes), 2)

        # Plot 1: Training loss and accuracy
        self.assertEqual(training_metrics_ax.get_title(), expected_training_metrics_ax_title)
        self.assertEqual(training_metrics_ax.get_xlabel(), expected_training_metrics_ax_xlabel)
        self.assertEqual(training_metrics_ax.get_ylabel(), expected_training_metrics_ax_ylabel)
        self.assertEqual(len(training_metrics_lines), expected_training_metrics_ax_number_of_lines)
        self.assertEqual(training_metrics_lines[0].get_xydata().tolist(),
                         expected_training_metrics_ax_xydata_first_line)
        self.assertEqual(training_metrics_lines[1].get_xydata().tolist(),
                         expected_training_metrics_ax_xydata_second_line)

        # Plot 2: Validation loss and accuracy
        self.assertEqual(validation_metrics_ax.get_title(), expected_validation_metrics_ax_title)
        self.assertEqual(validation_metrics_ax.get_xlabel(), expected_validation_metrics_ax_xlabel)
        self.assertEqual(validation_metrics_ax.get_ylabel(), expected_validation_metrics_ax_ylabel)
        self.assertEqual(len(validation_metrics_lines), expected_validation_metrics_ax_number_of_lines)
        self.assertEqual(validation_metrics_lines[0].get_xydata().tolist(),
                         expected_validation_metrics_ax_xydata_first_line)
        self.assertEqual(validation_metrics_lines[1].get_xydata().tolist(),
                         expected_validation_metrics_ax_xydata_second_line)

    def test_classification_training_plot_responsibility_chain_plot__sklearn_able_to_handle(self) -> None:
        """
        Tests ClassificationTrainingPlotResponsibilityChain's plot functionality with valid input.

        Verifies that the plot method correctly handles input data with the 'classification_learning_sklearn'
        key and generates a plot with the expected title, labels, and data series. Tests that
        the plot contains three lines representing currency prices, asset values, and the
        moving average.

        Asserts:
            The plot has the correct title, axis labels, and number of data series.
            The data points in each plot match the input data.
        """

        mocked_input_data = {
            'key': ClassificationLearningStrategyHandler.PLOTTING_KEYS[1] + "_" + SklearnModelAdapter.TAG,
            'plot_data': {
                'learning_curve_data_train_sizes': np.array([20, 40, 60, 80]),
                'learning_curve_data_train_scores': np.array([[0.9, 0.91], [0.92, 0.93], [0.94, 0.95], [0.96, 0.97]]),
                'learning_curve_data_valid_scores': np.array([[0.48, 0.49], [0.5, 0.51], [0.53, 0.54], [0.55, 0.56]]),
                'validation_data_score': 0.52
            }
        }

        # Plot 1: Training learning curve
        expected_training_learning_curve_ax_title = 'Training learning curve'
        expected_training_learning_curve_ax_xlabel = 'Training examples'
        expected_training_learning_curve_ax_ylabel = 'Accuracy'
        expected_training_learning_curve_ax_number_of_lines = 1
        learning_curve_data_train_sizes = mocked_input_data['plot_data']['learning_curve_data_train_sizes']
        learning_curve_data_train_scores = mocked_input_data['plot_data']['learning_curve_data_train_scores']
        expected_training_learning_curve_ax_xydata_first_line = [list(tup) for tup in
            list(zip(learning_curve_data_train_sizes, learning_curve_data_train_scores.mean(axis = 1)))]

        # Plot 2: Validation learning curve
        validation_data_score = mocked_input_data['plot_data']['validation_data_score']
        expected_validation_learning_curve_ax_title = f"Validation learning curve, score: {validation_data_score:.2f}"
        expected_validation_learning_curve_ax_xlabel = 'Validation examples'
        expected_validation_learning_curve_ax_ylabel = 'Accuracy'
        expected_validation_learning_curve_ax_number_of_lines = 1
        learning_curve_data_valid_scores = mocked_input_data['plot_data']['learning_curve_data_valid_scores']
        expected_validation_learning_curve_ax_xydata_first_line = [list(tup) for tup in
            list(zip(learning_curve_data_train_sizes, learning_curve_data_valid_scores.mean(axis = 1)))]

        logging.info("Plotting using provided data.")
        _ = self.__sut.plot(mocked_input_data)
        fig = plt.gcf()

        # Plot 1: Training learning curve
        training_learning_curve_ax = fig.axes[0]
        training_learning_curve_lines = training_learning_curve_ax.get_lines()

        # Plot 2: Validation learning curve
        validation_learning_curve_ax = fig.axes[1]
        validation_learning_curve_lines = validation_learning_curve_ax.get_lines()

        logging.info("Checking expected plots.")
        self.assertEqual(len(fig.axes), 2)

        # Plot 1: Training learning curve
        self.assertEqual(training_learning_curve_ax.get_title(), expected_training_learning_curve_ax_title)
        self.assertEqual(training_learning_curve_ax.get_xlabel(), expected_training_learning_curve_ax_xlabel)
        self.assertEqual(training_learning_curve_ax.get_ylabel(), expected_training_learning_curve_ax_ylabel)
        self.assertEqual(len(training_learning_curve_lines), expected_training_learning_curve_ax_number_of_lines)
        self.assertEqual(training_learning_curve_lines[0].get_xydata().tolist(),
                         expected_training_learning_curve_ax_xydata_first_line)

        # Plot 2: Validation learning curve
        self.assertEqual(validation_learning_curve_ax.get_title(), expected_validation_learning_curve_ax_title)
        self.assertEqual(validation_learning_curve_ax.get_xlabel(), expected_validation_learning_curve_ax_xlabel)
        self.assertEqual(validation_learning_curve_ax.get_ylabel(), expected_validation_learning_curve_ax_ylabel)
        self.assertEqual(len(validation_learning_curve_lines), expected_validation_learning_curve_ax_number_of_lines)
        self.assertEqual(validation_learning_curve_lines[0].get_xydata().tolist(),
                         expected_validation_learning_curve_ax_xydata_first_line)

    def test_classification_training_plot_responsibility_chain_plot__unable_to_handle(self) -> None:
        """
        Tests ClassificationTrainingPlotResponsibilityChain's plot functionality with invalid input.

        Verifies that the plot method correctly handles the case where the key does not
        match 'classification_training_tensorflow' nor 'classification_training_sklearn'.
        In this case, the handler should not process the request and should return None,
        indicating that the request should be passed to the next handler in the chain.

        Asserts:
            The method returns None when given an unrecognized key.
        """

        logging.info("Starting plot test.")
        mocked_input_data = {
            'key': 'unknown_plot_type',
            'plot_data': None
        }

        logging.info("Plotting using provided data.")
        result = self.__sut.plot(mocked_input_data)

        logging.info("Checking if plot is empty.")
        self.assertEqual(result, None)