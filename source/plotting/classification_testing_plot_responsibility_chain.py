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
        ax1.set_title(f"Confusion Matrix (Accuracy: {additional_report['accuracy']:.2%})")

        normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis = 1, keepdims = True)
        normalized_conf_matrix = np.round(np.nan_to_num(normalized_conf_matrix, nan = 0.0), 2)
        ax1.imshow(normalized_conf_matrix, interpolation = 'nearest', cmap = plt.cm.GnBu)

        tick_marks = np.arange(len(classes))
        ax1.set_xticks(tick_marks)
        ax1.set_yticks(tick_marks)
        ax1.set_xticklabels(shortened_classes_names)
        ax1.set_yticklabels(shortened_classes_names)
        ax1.set_xlabel('Predicted label')
        ax1.set_ylabel('True label')

        thresh = np.max(conf_matrix, axis = 1) / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax1.text(j, i - 0.1, format(conf_matrix[i, j], 'd'),
                        ha = "center", va = "center", fontsize = 10, weight = 'bold',
                        color = "white" if conf_matrix[i, j] > thresh[i] else "black")
                ax1.text(j, i + 0.15, f'{normalized_conf_matrix[i, j]:.2f}',
                        ha = "center", va = "center", fontsize = 8,
                        color = "white" if conf_matrix[i, j] > thresh[i] else "black")

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
        precision_bars = ax2.bar(tick_marks - shift, precision_scores, shift, label = 'Precision')
        recall_bars = ax2.bar(tick_marks, recall_scores, shift, label = 'Recall')
        f1_bars = ax2.bar(tick_marks + shift, f1_scores, shift, label = 'F1-score')

        for i, (precision_bar, recall_bar, f1_bar) in enumerate(zip(precision_bars, recall_bars, f1_bars)):
            ax2.text(precision_bar.get_x() + (precision_bar.get_width() / 2),
                     precision_bar.get_height() + 0.01 if precision_bar.get_height() < 0.9 else \
                        precision_bar.get_height() - 0.01, f'{precision_scores[i]:.3f}',
                    ha = 'center', va = 'bottom' if precision_bar.get_height() < 0.9 else 'top', rotation = 90,
                    fontsize = 8, weight = 'bold')
            ax2.text(recall_bar.get_x() + (recall_bar.get_width() / 2),
                     recall_bar.get_height() + 0.01 if recall_bar.get_height() < 0.9 else \
                        recall_bar.get_height() - 0.01, f'{recall_scores[i]:.3f}',
                    ha = 'center', va = 'bottom' if recall_bar.get_height() < 0.9 else 'top', rotation = 90,
                    fontsize = 8, weight = 'bold')
            ax2.text(f1_bar.get_x() + (f1_bar.get_width() / 2),
                     f1_bar.get_height() + 0.01 if f1_bar.get_height() < 0.9 else \
                        f1_bar.get_height() - 0.01, f'{f1_scores[i]:.3f}',
                    ha = 'center', va = 'bottom' if f1_bar.get_height() < 0.9 else 'top', rotation = 90,
                    fontsize = 8, weight = 'bold')

        ax2.set_title('Classification metrics by class')
        ax2.set_xticks(tick_marks)
        ax2.set_xticklabels(shortened_classes_names)
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Score')
        ax2.set_ylim([0, 1])
        ax2.legend(fontsize = 'x-small')

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
        ax3.legend(loc = "lower right", fontsize = 'x-small')
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
        precision_bars = ax4.bar(x - shift, precision_scores, shift, label = 'Precision')
        recall_bars = ax4.bar(x, recall_scores, shift, label = 'Recall')
        f1_bars = ax4.bar(x + shift, f1_scores, shift, label = 'F1-score')

        for i, (precision_bar, recall_bar, f1_bar) in enumerate(zip(precision_bars, recall_bars, f1_bars)):
            ax4.text(precision_bar.get_x() + (precision_bar.get_width() / 2),
                     precision_bar.get_height() + 0.01 if precision_bar.get_height() < 0.9 else \
                        precision_bar.get_height() - 0.01, f'{precision_scores[i]:.3f}',
                    ha = 'center', va = 'bottom' if precision_bar.get_height() < 0.9 else 'top', rotation = 90,
                    fontsize = 8, weight = 'bold')
            ax4.text(recall_bar.get_x() + (recall_bar.get_width() / 2),
                     recall_bar.get_height() + 0.01 if recall_bar.get_height() < 0.9 else \
                        recall_bar.get_height() - 0.01, f'{recall_scores[i]:.3f}',
                    ha = 'center', va = 'bottom' if recall_bar.get_height() < 0.9 else 'top', rotation = 90,
                    fontsize = 8, weight = 'bold')
            ax4.text(f1_bar.get_x() + (f1_bar.get_width() / 2),
                     f1_bar.get_height() + 0.01 if f1_bar.get_height() < 0.9 else \
                        f1_bar.get_height() - 0.01, f'{f1_scores[i]:.3f}',
                    ha = 'center', va = 'bottom' if f1_bar.get_height() < 0.9 else 'top', rotation = 90,
                    fontsize = 8, weight = 'bold')

        ax4.set_title('Macro avg and weighted avg')
        ax4.set_xticks(x)
        ax4.set_xticklabels(additional_labels)
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Score')
        ax4.set_ylim([0, 1])
        ax4.legend(fontsize = 'x-small')
        plt.tight_layout()

        return plt.gca()
