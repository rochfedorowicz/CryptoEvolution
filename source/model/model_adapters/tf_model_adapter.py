# model/model_adapters/tf_model_adapter.py

# global imports
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import to_categorical
from typing import Any, Callable, Optional

# local imports
from source.model import ModelAdapterBase

class TFModelAdapter(ModelAdapterBase):
    """
    Implements a model adapter for TensorFlow models. It provides methods for loading,
    saving, printing summaries, fitting, predicting, and retrieving the model.
    """

    # global class constants
    TAG: str = "tensorflow"

    # local constants
    __WEIGHTS_FILE_EXTENSION: str = ".h5"
    __OPTIMIZER: str = "adam"
    __LOSS: str = "categorical_crossentropy"
    __METRICS: list[str] = ["accuracy"]

    def __init__(self, model: Model, optimizer: Optional[Optimizer] = None,
                 loss: Optional[str] = None, metrics: Optional[list[str]] = None) -> None:
        """
        Class constructor. Initializes the model adapter with a TensorFlow model and optional parameters.

        Parameters:
            model (Model): The TensorFlow model to adapt.
            optimizer (Optional[Optimizer]): The optimizer to use for training the model.
                Defaults to Adam if not provided.
            loss (Optional[str]): The loss function to use for training the model.
                Defaults to categorical_crossentropy if not provided.
            metrics (Optional[list[str]]): The metrics to evaluate during training.
                Defaults to accuracy if not provided.
        """

        if optimizer is None:
            optimizer = self.__OPTIMIZER
        if loss is None:
            loss = self.__LOSS
        if metrics is None:
            metrics = self.__METRICS

        self.__model: Model = model
        self.__model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

        self.__adjust_data_func = lambda x, y = None: (
            np.expand_dims(np.array(x), axis = 1),
            to_categorical(np.array(y), num_classes = self.__model.output_shape[1]) if y is not None else None
        )

    def load_model(self, path: str) -> None:
        """
        Loads a TensorFlow model from a file.

        Parameters:
            path (str): The path to the model file.

        Raises:
            ValueError: If the path does not end with the expected weights file extension.
        """

        if self.__WEIGHTS_FILE_EXTENSION not in path:
            raise ValueError(f"Model path must end with '{self.__WEIGHTS_FILE_EXTENSION}'.")

        self.__model.load_weights(path)

    def save_model(self, path: str) -> None:
        """
        Saves a TensorFlow model to a file.

        Parameters:
            path (str): The path to the model file.

        Raises:
            ValueError: If the path does not end with the expected weights file extension.
        """

        if self.__WEIGHTS_FILE_EXTENSION not in path:
            raise ValueError(f"Model path must end with '{self.__WEIGHTS_FILE_EXTENSION}'.")

        self.__model.save_weights(path)

    def print_summary(self, print_function: Callable = print) -> None:
        """
        Prints a summary of the model's architecture.

        Parameters:
            print_function (Callable): The function to use for printing the summary.
        """

        self.__model.summary(print_fn = print_function)

    def fit(self, input_data: Any, output_data: Any, validation_data: Any,
            epochs: int, batch_size: int, callbacks: list[Callback], **kwargs) -> dict:
        """
        Fits the model to the provided input and output data.

        Parameters:
            input_data (Any): The input data for fitting the model.
            output_data (Any): The output data for fitting the model.
            validation_data (Any): The validation data for evaluating the model.
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size to use for training.
            callbacks (list[Callback]): The list of callbacks to use during training.
            (**kwargs): Additional keyword arguments for fitting the model.

        Returns:
            (dict): A dictionary containing the results of the fitting process.
        """

        validation_data = self.__adjust_data_func(validation_data[0], validation_data[1])
        input_data, output_data = self.__adjust_data_func(input_data, output_data)

        return self.__model.fit(input_data, output_data, epochs = epochs,
                                validation_data = validation_data, batch_size = batch_size,
                                callbacks = callbacks, **kwargs).history

    def predict(self, data: Any) -> dict:
        """
        Predicts probabilities for the output for the given input data.

        Parameters:
            data (Any): The input data for prediction.
        """

        data, _ = self.__adjust_data_func(data)
        return self.__model.predict(data)

    def get_model(self) -> Model:
        """
        Retrieves the underlying model.

        Returns:
            (Model): The underlying model instance.
        """

        return self.__model