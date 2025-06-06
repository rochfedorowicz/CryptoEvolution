# agent/strategies/classification_learning_agent.py

# global imports
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback
from typing import Any

# local imports
from source.agent import AgentBase, ClassificationTestable

class ClassificationLearningAgent(AgentBase, ClassificationTestable):
    """
    Implements a classification learning agent that can be trained and tested
    in a trading environment. It provides functionalities for fitting the model
    with classification data and making predictions.
    """

    def classification_fit(self, input_data: np.ndarray, output_data: np.ndarray,
                           validation_data: tuple[np.ndarray, np.ndarray], batch_size: int,
                           epochs: int, callbacks: list[Callback]) -> dict[str, Any]:
        """
        Fits the model to the classification data. Parameters passed to the model adapter
        are dynamically determined based on the model's requirements.

        Parameters:
            input_data (np.ndarray): The input data for training.
            output_data (np.ndarray): The output data for training.
            validation_data (tuple[np.ndarray, np.ndarray]): The validation data (input, output).
            batch_size (int): The batch size to be used during training.
            epochs (int): The number of epochs to train the model.
            callbacks (list[Callback]): A list of callbacks to be used during training.

        Returns:
            (dict[str, Any]): A dictionary containing the training history and other relevant information.
        """

        provided_input_params = {}
        provided_input_params['input_data'] = input_data
        provided_input_params['output_data'] = output_data
        provided_input_params['validation_data'] = validation_data
        provided_input_params['batch_size'] = batch_size
        provided_input_params['epochs'] = epochs
        provided_input_params['callbacks'] = callbacks

        parameters_needed_for_fitting = self._model_adapter.report_parameters_needed_for_fitting()
        kwargs = {}
        for parameter in parameters_needed_for_fitting:
            kwargs[parameter] = provided_input_params.get(parameter, None)

        return self._model_adapter.fit(**kwargs)

    def classify(self, data: pd.DataFrame) -> list[list[float]]:
        """
        Classifies the input data using the trained model.

        Parameters:
            data (pd.DataFrame): The input data to be classified.

        Returns:
            (list[list[float]]): The predicted class probabilities for each input sample.
        """

        return self._model_adapter.predict(data)

    def get_model_adapter_tag(self) -> str:
        """
        Returns the tag of the model adapter.

        Returns:
            (str): The tag of the model adapter.
        """

        return type(self._model_adapter).TAG
