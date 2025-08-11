# tests/plotting/test_classification_testing_plot_responsibility_chain.py

# global imports
import logging
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from ddt import ddt
from unittest import TestCase

# local imports
from source.agent import ClassificationTestingStrategyHandler
from source.plotting import ClassificationTestingPlotResponsibilityChain

@ddt
class ClassificationTestingPlotResponsibilityChainTestCase(TestCase):
    """
    Test case for ClassificationTestingPlotResponsibilityChain class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: ClassificationTestingPlotResponsibilityChain = ClassificationTestingPlotResponsibilityChain()

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")
        plt.close('all')

    def test_classification_testing_plot_responsibility_chain_plot__able_to_handle(self) -> None:
        """
        Tests ClassificationTestingPlotResponsibilityChain's plot functionality with valid input.

        Verifies that the plot method correctly handles input data with the 'classification_testing'
        key and generates a plot with the expected title, labels, and data series. Tests that
        the plot contains three lines representing currency prices, asset values, and the
        moving average.

        Asserts:
            The plot has the correct title, axis labels, and number of data series.
            The data points in each plot match the input data.
        """

        mocked_input_data = {
            'key': ClassificationTestingStrategyHandler.PLOTTING_KEY,
            'plot_data': {
                'confusion_matrix': np.array([[6, 1],  # 6 true negatives, 1 false positive
                                             [2, 1]]), # 2 false negatives, 1 true positive
                'classification_report': {
                    'class_0': {'precision': 0.75, 'recall': 0.86, 'f1-score': 0.80, 'support': 7},
                    'class_1': {'precision': 0.50, 'recall': 0.33, 'f1-score': 0.40, 'support': 3},
                    'accuracy': 0.70,
                    'macro avg': {'precision': 0.625, 'recall': 0.595, 'f1-score': 0.60, 'support': 10},
                    'weighted avg': {'precision': 0.675, 'recall': 0.70, 'f1-score': 0.68, 'support': 10}
                },
                'prediction_probabilities': np.array([
                    [0.85, 0.15],  # Sample 1 (predicted class 0) -> true negative
                    [0.90, 0.10],  # Sample 2 (predicted class 0) -> true negative
                    [0.78, 0.22],  # Sample 3 (predicted class 0) -> true negative
                    [0.95, 0.05],  # Sample 4 (predicted class 0) -> true negative
                    [0.60, 0.40],  # Sample 5 (predicted class 0) -> true negative
                    [0.82, 0.18],  # Sample 6 (predicted class 0) -> true negative
                    [0.70, 0.30],  # Sample 7 (predicted class 0) -> false negative
                    [0.25, 0.75],  # Sample 8 (predicted class 1) -> false positive
                    [0.60, 0.40],  # Sample 9 (predicted class 0) -> false negative
                    [0.20, 0.80],  # Sample 10 (predicted class 1) -> true positive
                ]),
                'true_labels': np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1]),
            }
        }

        # Plot 1: Confusion Matrix as a heatmap
        expected_accuracy = mocked_input_data['plot_data']['classification_report']['accuracy']
        expected_confusion_matrix_ground_truth = mocked_input_data['plot_data']['confusion_matrix']
        expected_confusion_matrix_ax_title = f"Confusion Matrix (Accuracy: {expected_accuracy:.2%})"
        expected_confusion_matrix_ax_xlabel = "Predicted label"
        expected_confusion_matrix_ax_ylabel = "True label"

        # Plot 2: Precision, Recall, F1 Score Bar Chart
        expected_classification_metrics_ax_title = "Classification metrics by class"
        expected_classification_metrics_ax_xlabel = "Classes"
        expected_classification_metrics_ax_ylabel = "Score"
        expected_classification_metrics_ax_legend_texts = ['Precision', 'Recall', 'F1-score']
        expected_classification_metrics_bar_heights = [
            mocked_input_data['plot_data']['classification_report']['class_0']['precision'],
            mocked_input_data['plot_data']['classification_report']['class_1']['precision'],
            mocked_input_data['plot_data']['classification_report']['class_0']['recall'],
            mocked_input_data['plot_data']['classification_report']['class_1']['recall'],
            mocked_input_data['plot_data']['classification_report']['class_0']['f1-score'],
            mocked_input_data['plot_data']['classification_report']['class_1']['f1-score']
        ]

        # Plot 3: OvR-ROC curves
        expected_roc_curve_ax_title = 'One-vs-Rest ROC curves'
        expected_roc_curve_ax_xlabel = 'False positive rate'
        expected_roc_curve_ax_ylabel = 'True positive rate'
        expected_roc_curve_ax_number_of_lines = 3

        # Plot 4: Macro avg and weighted avg
        expected_cumulative_metrics_ax_title = 'Macro avg and weighted avg'
        expected_cumulative_metrics_ax_xlabel = 'Metrics'
        expected_cumulative_metrics_ax_ylabel = 'Score'
        expected_cumulative_metrics_ax_legend_texts = ['Precision', 'Recall', 'F1-score']
        expected_cumulative_metrics_ax_bar_heights = [
            mocked_input_data['plot_data']['classification_report']['macro avg']['precision'],
            mocked_input_data['plot_data']['classification_report']['weighted avg']['precision'],
            mocked_input_data['plot_data']['classification_report']['macro avg']['recall'],
            mocked_input_data['plot_data']['classification_report']['weighted avg']['recall'],
            mocked_input_data['plot_data']['classification_report']['macro avg']['f1-score'],
            mocked_input_data['plot_data']['classification_report']['weighted avg']['f1-score']
        ]

        logging.info("Plotting using provided data.")
        _ = self.__sut.plot(mocked_input_data)
        fig = plt.gcf()

        # Plot 1: Confusion Matrix as a heatmap
        confusion_matrix_ax = fig.axes[0]
        drawn_matrix_ground_truth = np.zeros((2, 2), dtype = int)
        for text in confusion_matrix_ax.texts:
            if text.get_text().isdigit():
                col, row = map(float, text.get_position())
                drawn_matrix_ground_truth[int(np.ceil(row)), int(np.ceil(col))] = int(text.get_text())

        # Plot 2: Precision, Recall, F1 Score Bar Chart
        classification_metrics_ax = fig.axes[1]
        drawn_bars_classification_metrics = \
            [child for child in classification_metrics_ax.get_children()[:-1] \
             if isinstance(child, matplotlib.patches.Rectangle)]
        legend_texts_classification_metrics = \
            [text.get_text() for text in classification_metrics_ax.get_legend().get_texts()]

        # Plot 3: OvR-ROC curves
        roc_curve_ax = fig.axes[2]
        roc_lines = roc_curve_ax.get_lines()

        # Plot 4: Macro avg and weighted avg
        cumulative_metrics_ax = fig.axes[3]
        drawn_bars_cumulative_metrics = \
            [child for child in cumulative_metrics_ax.get_children()[:-1] \
             if isinstance(child, matplotlib.patches.Rectangle)]
        legend_texts_cumulative_metrics = \
            [text.get_text() for text in cumulative_metrics_ax.get_legend().get_texts()]

        logging.info("Checking expected plots.")
        self.assertEqual(len(fig.axes), 4)

        # Plot 1: Confusion Matrix as a heatmap
        self.assertEqual(confusion_matrix_ax.get_title(), expected_confusion_matrix_ax_title)
        self.assertEqual(confusion_matrix_ax.get_xlabel(), expected_confusion_matrix_ax_xlabel)
        self.assertEqual(confusion_matrix_ax.get_ylabel(), expected_confusion_matrix_ax_ylabel)
        self.assertTrue(isinstance(confusion_matrix_ax.get_children()[0], plt.cm.ScalarMappable))
        self.assertEqual(drawn_matrix_ground_truth.tolist(), expected_confusion_matrix_ground_truth.tolist())

        # Plot 2: Precision, Recall, F1 Score Bar Chart
        self.assertEqual(classification_metrics_ax.get_title(), expected_classification_metrics_ax_title)
        self.assertEqual(classification_metrics_ax.get_xlabel(), expected_classification_metrics_ax_xlabel)
        self.assertEqual(classification_metrics_ax.get_ylabel(), expected_classification_metrics_ax_ylabel)
        self.assertEqual(len(drawn_bars_classification_metrics), 6)
        self.assertEqual([bar.get_height() for bar in drawn_bars_classification_metrics],
                         expected_classification_metrics_bar_heights)
        self.assertEqual(legend_texts_classification_metrics, expected_classification_metrics_ax_legend_texts)

        # Plot 3: OvR-ROC curves
        self.assertEqual(roc_curve_ax.get_title(), expected_roc_curve_ax_title)
        self.assertEqual(roc_curve_ax.get_xlabel(), expected_roc_curve_ax_xlabel)
        self.assertEqual(roc_curve_ax.get_ylabel(), expected_roc_curve_ax_ylabel)
        self.assertEqual(len(roc_lines), expected_roc_curve_ax_number_of_lines)

        # Plot 4: Macro avg and weighted avg
        self.assertEqual(cumulative_metrics_ax.get_title(), expected_cumulative_metrics_ax_title)
        self.assertEqual(cumulative_metrics_ax.get_xlabel(), expected_cumulative_metrics_ax_xlabel)
        self.assertEqual(cumulative_metrics_ax.get_ylabel(), expected_cumulative_metrics_ax_ylabel)
        self.assertEqual(len(drawn_bars_cumulative_metrics), 6)
        self.assertEqual([bar.get_height() for bar in drawn_bars_cumulative_metrics],
                         expected_cumulative_metrics_ax_bar_heights)
        self.assertEqual(legend_texts_cumulative_metrics, expected_cumulative_metrics_ax_legend_texts)

    def test_classification_testing_plot_responsibility_chain_plot__unable_to_handle(self) -> None:
        """
        Tests ClassificationTestingPlotResponsibilityChain's plot functionality with invalid input.

        Verifies that the plot method correctly handles the case where the key does not
        match 'classification_testing'. In this case, the handler should not process the request
        and should return None, indicating that the request should be passed to the next
        handler in the chain.

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