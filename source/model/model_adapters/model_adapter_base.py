# model/model_adapters/model_adapter_base.py

# global imports
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable

# local imports

class ModelAdapterBase(ABC):
    """
    Implements a base class for model adapters. It provides an interface for loading,
    saving, printing summaries, fitting, predicting, and retrieving the model.
    """

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Loads a model from the specified path.

        Parameters:
            path (str): The path to the model file.
        """

        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Saves the model to the specified path.

        Parameters:
            path (str): The path to the model file.
        """

        pass

    @abstractmethod
    def print_summary(self, print_function: Callable = print) -> None:
        """
        Prints a summary of the model's architecture and parameters.

        Parameters:
            print_function (Callable): The function to use for printing the summary.
                Defaults to the built-in print function.
        """

        pass

    @abstractmethod
    def fit(self, input_data: Any, output_data: Any, **kwargs) -> Any:
        """
        Fits the model to the provided input and output data.

        Parameters:
            input_data (Any): The input data for fitting the model.
            output_data (Any): The output data for fitting the model.
            **kwargs: Additional keyword arguments for fitting the model.

        Returns:
            (Any): The result of the fitting process, which may vary depending on the model.
        """

        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """
        Predicts the output for the given input data.

        Parameters:
            data (Any): The input data for prediction.

        Returns:
            (Any): The predicted output data.
        """

        pass

    @abstractmethod
    def get_model(self) -> Any:
        """
        Retrieves the underlying model.

        Returns:
            (Any): The underlying model instance.
        """

        pass

    def report_parameters_needed_for_fitting(self) -> list[str]:
        """
        Reports the parameters needed for fitting the model.

        Returns:
            (list[str]): A list of parameter names that are required for fitting.
        """

        fit_function_params = dict(inspect.signature(self.fit).parameters)

        return [name for name, param in fit_function_params.items()
                 if param.default is param.empty and param.kind not in [param.VAR_KEYWORD, param.VAR_POSITIONAL]]