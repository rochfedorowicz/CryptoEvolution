# model/model_adapters/sklearn_model_adapter.py

# global imports
import joblib
import numpy as np
import os
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import learning_curve, StratifiedKFold
from typing import Any, Callable

# local imports
from source.model import ModelAdapterBase

class SklearnModelAdapter(ModelAdapterBase):
    """
    Implements a model adapter for scikit-learn models. It provides methods for loading,
    saving, printing summaries, fitting, predicting, and retrieving the model.
    """

    # global class constants
    TAG: str = "sklearn"

    # local constants
    __MODEL_FILE_EXTENSION: str = ".pkl"

    def __init__(self, model: BaseEstimator, should_compute_learning_curve: bool = True) -> None:
        """
        Initializes the SklearnModelAdapter with a scikit-learn model.

        Parameters:
            model (BaseEstimator): The scikit-learn model to adapt.
            should_compute_learning_curve (bool): Flag indicating whether to compute the
                learning curve. Defaults to True.
        """

        self.__model: BaseEstimator = model
        self.__should_compute_learning_curve: bool = should_compute_learning_curve

    def load_model(self, path: str) -> None:
        """
        Loads a scikit-learn model from a file.

        Parameters:
            path (str): The path to the model file.

        Raises:
            ValueError: If the path does not end with the expected model file extension.
            FileNotFoundError: If the model file does not exist at the specified path.
        """

        if not path.endswith(self.__MODEL_FILE_EXTENSION):
            raise ValueError(f"Model path must end with '{self.__MODEL_FILE_EXTENSION}'.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        loaded_model = joblib.load(path)
        self.__model = loaded_model

    def save_model(self, path: str) -> None:
        """
        Saves a scikit-learn model to a file.

        Parameters:
            path (str): The path to the model file.

        Raises:
            ValueError: If the path does not end with the expected model file extension.
        """

        if not path.endswith(self.__MODEL_FILE_EXTENSION):
            raise ValueError(f"Model path must end with '{self.__MODEL_FILE_EXTENSION}'.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok = True)
        joblib.dump(self.__model, path)

    def print_summary(self, print_function: Callable = print) -> None:
        """
        Prints a summary of the model's name and parameters.
        Uses the provided print function to output the summary.

        Parameters:
            print_function (Callable): The function to use for printing the summary.
        """

        print_function(f"{'-'*80}")
        print_function(f"Model Summary: {type(self.__model).__name__}")
        print_function(f"{'-'*80}")

        print_function("Model Parameters:")
        params = self.__model.get_params()
        for key, value in sorted(params.items()):
            print_function(f"    {key}: {value}")
        print_function(f"{'-'*80}")

    def fit(self, input_data: Any, output_data: Any, validation_data: Any, **kwargs) -> dict:
        """
        Fits the model to the provided input and output data.

        Parameters:
            input_data (Any): The input data for fitting the model.
            output_data (Any): The output data for fitting the model.
            validation_data (Any): The validation data for evaluating the model.
            **kwargs: Additional keyword arguments for fitting the model.

        Returns:
            (dict): A dictionary containing the results of the fitting process.
        """

        summary_data = {}
        cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
        if self.__should_compute_learning_curve:
            is_verbose = self.__model.get_params().get('verbose', False)
            self.__model.set_params(verbose = False)

            train_sizes = np.linspace(0.1, 1.0, 5)
            train_sizes_abs, train_scores, valid_scores = learning_curve(
                self.__model, input_data, output_data,
                train_sizes = train_sizes, cv = cv,
                scoring = 'accuracy', n_jobs = -1
            )

            self.__model.set_params(verbose = is_verbose)
            summary_data = {
                "learning_curve_data_train_sizes": train_sizes_abs,
                "learning_curve_data_train_scores": train_scores,
                "learning_curve_data_valid_scores": valid_scores
            }

        if not hasattr(self.__model, "predict_proba"):
            self.__model = CalibratedClassifierCV(self.__model, cv = cv)
        self.__model.fit(input_data, output_data, **kwargs)
        summary_data["validation_data_score"] = self.__model.score(*validation_data)

        return summary_data

    def predict(self, data: Any) -> dict:
        """
        Predicts probabilities for the output for the given input data.

        Parameters:
            data (Any): The input data for prediction.

        Returns:
            (dict): The predicted output data.
        """

        return self.__model.predict_proba(data)

    def get_model(self) -> BaseEstimator:
        """
        Retrieves the underlying model.

        Returns:
            (BaseEstimator): The underlying model instance.
        """

        return self.__model