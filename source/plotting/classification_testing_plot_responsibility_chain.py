# plotting/classification_testing_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import RocCurveDisplay

# local imports
from source.agent import ClassificationTestingStrategyHandler
from source.plotting import PlotResponsibilityChainBase

class ClassificationTestingPlotResponsibilityChain(PlotResponsibilityChainBase):
    """
    Implements a plotting responsibility chain for classification testing results.
    It implements the _can_plot and _plot methods to visualize confusion matrices,
    classification reports, and ROC curves.
    """

    # local constants
    __ADDITIONAL_REPORT_LABELS = ["accuracy", "macro avg", "weighted avg"]

    def _can_plot(self, key: str) -> bool:
        """
        Checks if the plot can be generated for the given key.

        Parameters:
            key (str): The key to check.

        Returns:
            (bool): True if the plot can be generated, False otherwise.
        """

        return key == ClassificationTestingStrategyHandler.PLOTTING_KEY

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates the classification testing plot based on the provided data.

        Parameters:
            plot_data (dict): The data to be plotted.

        Returns:
            (plt.Axes): The axes object containing the plot.
        """

        conf_matrix = plot_data.get("confusion_matrix", None)
        class_report = plot_data.get("classification_report", None)
        prediction_probabilities = plot_data.get("prediction_probabilities", None)
        true_labels = plot_data.get("true_labels", None)

        if conf_matrix is None or class_report is None or prediction_probabilities is None or true_labels is None:
            logging.warning(f"Insufficient data for plotting results under key: {ClassificationTestingStrategyHandler.PLOTTING_KEY}.")
            plt.text(0.5, 0.5, "Insufficient data for plotting",
                     ha = 'center', va = 'center', fontsize = 12)
            return plt.gca()

        additional_report = {}
        for additional_label in self.__ADDITIONAL_REPORT_LABELS:
            if additional_label in class_report:
                additional_report[additional_label] = class_report.pop(additional_label)

        fig = plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        gs = GridSpec(2, 2, figure = fig)
        classes = list(class_report.keys())
        shortened_classes_names = [class_name[:3] for class_name in classes]

        # Plot 1: Confusion Matrix as a heatmap
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(conf_matrix, interpolation = 'nearest', cmap = plt.cm.YlOrRd)
        ax1.set_title(f"Confusion Matrix (Accuracy: {additional_report['accuracy']:.2%})")

        tick_marks = np.arange(len(classes))
        ax1.set_xticks(tick_marks)
        ax1.set_yticks(tick_marks)
        ax1.set_xticklabels(shortened_classes_names)
        ax1.set_yticklabels(shortened_classes_names)
        ax1.set_xlabel('Predicted label')
        ax1.set_ylabel('True label')

        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax1.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")

        # Plot 2: Precision, Recall, F1 Score Bar Chart
        ax2 = plt.subplot(gs[1, 0])
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for metrics_dict in class_report.values():
            precision_scores.append(metrics_dict["precision"])
            recall_scores.append(metrics_dict["recall"])
            f1_scores.append(metrics_dict["f1-score"])

        shift = 0.2
        ax2.bar(tick_marks - shift, precision_scores, shift, label = 'Precision')
        ax2.bar(tick_marks, recall_scores, shift, label = 'Recall')
        ax2.bar(tick_marks + shift, f1_scores, shift, label = 'F1-score')

        ax2.set_title('Classification metrics by class')
        ax2.set_xticks(tick_marks)
        ax2.set_xticklabels(shortened_classes_names)
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Score')
        ax2.set_ylim([0, 1])
        ax2.legend()

        # Plot 3: OvR-ROC curves
        ax3 = plt.subplot(gs[0, 1])

        for i, class_name in enumerate(classes):
            y_true_class_binary = (true_labels == i).astype(int)
            y_score = prediction_probabilities[:, i]
            RocCurveDisplay.from_predictions(y_true_class_binary, y_score, name = f"{class_name}",
                                             ax = ax3, plot_chance_level = (i == len(classes) - 1))

        ax3.set_title('One-vs-Rest ROC curves')
        ax3.set_xlabel('False positive rate')
        ax3.set_ylabel('True positive rate')
        ax3.grid(alpha = 0.3)
        ax3.legend(loc = "lower right", fontsize = 'small')
        plt.tight_layout()

        # Plot 4: Macro avg and weighted avg
        ax4 = plt.subplot(gs[1, 1])
        additional_labels = list(additional_report.keys())[1:]
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for metrics in additional_report.values():
            if isinstance(metrics, dict):
                precision_scores.append(metrics['precision'])
                recall_scores.append(metrics['recall'])
                f1_scores.append(metrics['f1-score'])

        x = np.arange(len(additional_labels))
        ax4.bar(x - shift, precision_scores, shift, label = 'Precision')
        ax4.bar(x, recall_scores, shift, label = 'Recall')
        ax4.bar(x + shift, f1_scores, shift, label = 'F1-score')

        ax4.set_title('Macro avg and weighted avg')
        ax4.set_xticks(x)
        ax4.set_xticklabels(additional_labels)
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Score')
        ax4.set_ylim([0, 1])
        ax4.legend()
        plt.tight_layout()

        return plt.gca()
