# agent/strategies/learning_strategy_handler_base.py

# global imports
from abc import ABC, abstractmethod
from tensorflow.keras.callbacks import Callback
from typing import Any

# local imports
from source.agent import AgentBase
from source.environment import TradingEnvironment
from source.model import BluePrintBase

class LearningStrategyHandlerBase(ABC):
    """
    Implements a base class for learning strategy handlers. It provides an interface
    for creating agents, fitting them to the environment, and providing parameters
    needed for the model blueprint instantiation.
    """

    # global class constants
    PLOTTING_KEY: str = 'asset_price_movement_summary'

    def __init__(self) -> None:
        """
        Class constructor. Initializes the parameter provision callbacks.
        """

        self.__parameter_provision_callbacks = {
            'input_shape': self._provide_input_shape,
            'output_length': self._provide_output_length,
            'spatial_data_shape': self._provide_spatial_data_shape
        }

    @abstractmethod
    def create_agent(self, model_blue_print: BluePrintBase,
                     trading_environment: TradingEnvironment) -> AgentBase:
        """
        Creates an agent based on the provided model blueprint and trading environment.

        Parameters:
            model_blue_print (BluePrintBase): The model blueprint to be used for agent creation.
            trading_environment (TradingEnvironment): The trading environment in which the agent will operate.

        Returns:
            (AgentBase): An instance of the agent created using the model blueprint and trading environment.
        """

        pass

    @abstractmethod
    def fit(self, agent: AgentBase, environment: TradingEnvironment, nr_of_steps: int, nr_of_episodes: int,
            callbacks: list[Callback]) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Fits the agent to the environment using the specified number of steps and episodes. It also collects data
        for plotting summary statistics.

        Parameters:
            agent (AgentBase): The agent to be fitted to the environment.
            environment (TradingEnvironment): The trading environment in which the agent will be trained.
            nr_of_steps (int): The number of steps to be taken during training.
            nr_of_episodes (int): The number of episodes to be run during training.
            callbacks (list[Callback]): A list of Keras callbacks to be used during training.

        Returns:
            (tuple[list[str], list[dict[str, Any]]]): A tuple containing a list of keys and a list of dictionaries
            with the data collected during training.
        """

        data = {}

        environment.set_mode(TradingEnvironment.TEST_MODE)
        data['test_part_price_movement'] = environment.get_data_for_iteration(['close'])
        data['test_part_volatility'] = environment.get_data_for_iteration(['volatility'])

        environment.set_mode(TradingEnvironment.TRAIN_MODE)
        data['train_part_price_movement'] = environment.get_data_for_iteration(['close'])
        data['train_part_volatility'] = environment.get_data_for_iteration(['volatility'])

        return [LearningStrategyHandlerBase.PLOTTING_KEY], [data]

    def _provide_required_parameter(self, parameter: str, environment: TradingEnvironment) -> Any:
        """
        Provides the required parameter for the given environment.

        Parameters:
            parameter (str): The name of the parameter to be provided.
            environment (TradingEnvironment): The trading environment from which the parameter is to be provided.

        Raises:
            ValueError: If the parameter is not supported by this environment configuration.

        Returns:
            (Any): The value of the requested parameter.
        """

        if parameter in self.__parameter_provision_callbacks:
            return self.__parameter_provision_callbacks[parameter](environment)
        else:
            raise ValueError(f"Parameter '{parameter}' is not supported to provided by this environment configuration.")

    @abstractmethod
    def _provide_input_shape(self, environment: TradingEnvironment) -> tuple[int, int]:
        """
        Provides the input shape for the given environment.

        Parameters:
            environment (TradingEnvironment): The trading environment for which the input shape is to be provided.

        Returns:
            (tuple[int, int]): The input shape as a tuple.
        """

        pass

    @abstractmethod
    def _provide_output_length(self, environment: TradingEnvironment) -> int:
        """
        Provides the output length for the given environment.

        Parameters:
            environment (TradingEnvironment): The trading environment for which the output length is to be provided.

        Returns:
            (int): The output length.
        """

        pass

    @abstractmethod
    def _provide_spatial_data_shape(self, environment: TradingEnvironment) -> tuple[int, int]:
        """
        Provides the spatial data shape for the given environment.

        Parameters:
            environment (TradingEnvironment): The trading environment for which the spatial data shape is to be provided.

        Returns:
            (tuple[int, int]): The spatial data shape as a tuple.
        """

        pass
