# agent/agents/agent_base.py

# global imports
from typing import Callable, Optional

# local imports
from source.model import ModelAdapterBase

class AgentBase():
    """
    Implements base class for agents that can be trained and tested in a trading environment.
    It provides basic functionalities such as loading and saving models,printing model summaries.
    """

    def __init__(self, model_adapter: ModelAdapterBase) -> None:
        """
        Class constructor. Initializes the agent with the given model adapter.

        Parameters:
            model_adapter (ModelAdapterBase): The model adapter to be used for the agent.
        """

        self._model_adapter: ModelAdapterBase = model_adapter

    def load_model(self, model_path: str) -> None:
        """
        Loads the model from the specified path.

        Parameters:
            model_path (str): The path to the model file.
        """

        self._model_adapter.load_model(model_path)

    def save_model(self, model_path: str) -> None:
        """
        Saves the model to the specified path.

        Parameters:
            model_path (str): The path to the model file.
        """

        self._model_adapter.save_model(model_path)

    def print_summary(self, print_function: Optional[Callable] = print) -> None:
        """
        Prints a summary of the model architecture and parameters.

        Parameters:
            print_function (Optional[Callable]): A function to print the summary. Defaults to print.
        """

        self._model_adapter.print_summary(print_function)