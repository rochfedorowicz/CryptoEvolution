# plotting/classification_training_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
from enum import Enum
from matplotlib.gridspec import GridSpec

# local imports
from source.agent import ClassificationLearningStrategyHandler
from source.model import SklearnModelAdapter, TFModelAdapter
from source.plotting import PlotResponsibilityChainBase

class ClassificationTrainingPlotResponsibilityChain(PlotResponsibilityChainBase):
    """
    Implements a plotting responsibility chain for classification training results.
    It implements the _can_plot and _plot methods to visualize training loss and accuracy.
    """

    class PlottingMode(Enum):
        """
        Enumeration for the different plotting modes.
        It defines the modes for TensorFlow and Scikit-learn models.
        """

        UNSUPPORTED = "unsupported"
        TF = TFModelAdapter.TAG
        SKLEARN = SklearnModelAdapter.TAG

    def __init__(self) -> None:
        """
        Class constructor. Initializes the plotting responsibility chain with an unsupported mode.
        """

        self.__mode: ClassificationTrainingPlotResponsibilityChain.PlottingMode = ClassificationTrainingPlotResponsibilityChain.PlottingMode.UNSUPPORTED

    def _can_plot(self, key: str) -> bool:
        """
        Checks if the plot can be generated for the given key.

        Parameters:
            key (str): The key to check.

        Returns:
            (bool): True if the plot can be generated, False otherwise.
        """

        if TFModelAdapter.TAG in key or SklearnModelAdapter.TAG in key:
            self.__mode = ClassificationTrainingPlotResponsibilityChain.PlottingMode(key.split("_")[-1])

            return ClassificationLearningStrategyHandler.PLOTTING_KEYS[1] in key
        else:
            self.__mode = ClassificationTrainingPlotResponsibilityChain.PlottingMode.UNSUPPORTED

            return False

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates the plot based on the current mode and provided data.

        Parameters:
            plot_data (dict): The data to be plotted.

        Returns:
            (plt.Axes): The axes object containing the plot.
        """

        if self.__mode == ClassificationTrainingPlotResponsibilityChain.PlottingMode.TF:
            return self.__plot_tf(plot_data)
        elif self.__mode == ClassificationTrainingPlotResponsibilityChain.PlottingMode.SKLEARN:
            return self.__plot_sklearn(plot_data)
        else:
            raise ValueError(f"Unsupported plotting mode: {self.__mode}. Expected 'tensorflow' or 'sklearn'.")

    def __plot_tf(self, plot_data: dict) -> plt.Axes:
        """
        Generates the TensorFlow training plot.

        Parameters:
            plot_data (dict): The data to be plotted.

        Returns:
            (plt.Axes): The axes object containing the plot.
        """

        loss = plot_data.get("loss", None)
        accuracy = plot_data.get("accuracy", None)
        val_loss = plot_data.get("val_loss", None)
        val_accuracy = plot_data.get("val_accuracy", None)

        if loss is None or accuracy is None or val_loss is None or val_accuracy is None:
            logging.warning(f"Insufficient data for plotting results under key: {ClassificationLearningStrategyHandler.PLOTTING_KEYS[1]}.")
            plt.text(0.5, 0.5, "Insufficient data for plotting",
                     ha = 'center', va = 'center', fontsize = 12)
            return plt.gca()

        fig = plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        gs = GridSpec(2, 1, figure = fig)

        # Plot 1: Training loss and accuracy
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title("Training loss and accuracy")
        ax1.plot(loss, label = 'Loss')
        ax1.plot(accuracy, label = 'Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Value')
        ax1.legend(loc = 'upper left')

        # Plot 2: Validation loss and accuracy
        ax2 = plt.subplot(gs[1, 0])
        ax2.set_title("Validation loss and accuracy")
        ax2.plot(val_loss, label = 'Validation Loss')
        ax2.plot(val_accuracy, label = 'Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.legend(loc = 'upper left')
        plt.tight_layout()

        return plt.gca()

    def __plot_sklearn(self, plot_data: dict) -> plt.Axes:
        """
        Generates the Scikit-learn training plot.

        Parameters:
            plot_data (dict): The data to be plotted.

        Returns:
            (plt.Axes): The axes object containing the plot.
        """

        learning_curve_data_train_sizes = plot_data.get("learning_curve_data_train_sizes", None)
        learning_curve_data_train_scores = plot_data.get("learning_curve_data_train_scores", None)
        learning_curve_data_valid_scores = plot_data.get("learning_curve_data_valid_scores", None)
        validation_data_score = plot_data.get("validation_data_score", None)

        if learning_curve_data_train_sizes is None or learning_curve_data_train_scores is None \
            or learning_curve_data_valid_scores is None or validation_data_score is None:
            logging.warning(f"Insufficient data for plotting results under key: {ClassificationLearningStrategyHandler.PLOTTING_KEYS[1]}.")
            plt.text(0.5, 0.5, "Insufficient data for plotting",
                     ha = 'center', va = 'center', fontsize = 12)
            return plt.gca()

        fig = plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        gs = GridSpec(2, 1, figure = fig)

        train_scores_mean = learning_curve_data_train_scores.mean(axis = 1)
        train_scores_std = learning_curve_data_train_scores.std(axis = 1)
        valid_scores_mean = learning_curve_data_valid_scores.mean(axis = 1)
        valid_scores_std = learning_curve_data_valid_scores.std(axis = 1)

        # Plot 1: Training learning curve
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title("Training learning curve")
        ax1.fill_between(learning_curve_data_train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha = 0.1, color = "r")
        ax1.plot(learning_curve_data_train_sizes, train_scores_mean, 'o-', color = "r", label = "Training score")
        ax1.set_xlabel('Training examples')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc = 'upper left')

        # Plot 2: Validation learning curve
        ax2 = plt.subplot(gs[1, 0])
        ax2.set_title(f"Validation learning curve, score: {validation_data_score:.2f}")
        ax2.fill_between(learning_curve_data_train_sizes, valid_scores_mean - valid_scores_std,
                        valid_scores_mean + valid_scores_std, alpha = 0.1, color = "g")
        ax2.plot(learning_curve_data_train_sizes, valid_scores_mean, 'o-', color = "g", label = "Cross-validation score")
        ax2.set_xlabel('Validation examples')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc = 'upper left')
        plt.tight_layout()

        return plt.gca()
