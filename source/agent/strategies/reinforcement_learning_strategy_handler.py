# agent/strategies/reinforcement_learning_strategy_handler.py

# global imports
from rl.policy import BoltzmannQPolicy, Policy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam, Optimizer
from typing import Any

# local imports
from source.agent import AgentBase, LearningStrategyHandlerBase, ReinforcementLearningAgent
from source.environment import TradingEnvironment
from source.model import BluePrintBase, TFModelAdapter

class ReinforcementLearningStrategyHandler(LearningStrategyHandlerBase):
    """
    Implements a reinforcement learning strategy handler. It provides methods for creating
    reinforcement learning agents, fitting them to the trading environment, and providing
    parameters needed for model instantiation coherent with reinforcement learning.
    """

    # global class constants
    PLOTTING_KEY: str = 'reinforcement_learning'

    def __init__(self, policy: Policy = BoltzmannQPolicy(),
                 optimizer: Optimizer = Adam(learning_rate = 0.001)) -> None:
        """
        Class constructor. Initializes the policy and optimizer for the reinforcement learning agent.

        Parameters:
            policy (Policy): The policy to be used by the reinforcement learning agent.
            optimizer (Optimizer): The optimizer to be used for training the reinforcement learning agent.
        """

        super().__init__()
        self.__policy: Policy = policy
        self.__optimizer: Optimizer = optimizer

    def create_agent(self, model_blue_print: BluePrintBase,
                     trading_environment: TradingEnvironment) -> AgentBase:
        """
        Creates a reinforcement learning agent. It dynamically determines the parameters
        needed for instantiation based on the model blueprint.

        Parameters:
            model_blue_print (BluePrintBase): The model blueprint to be used for agent creation.
            trading_environment (TradingEnvironment): The trading environment in which the agent will operate.

        Raises:
            TypeError: If the model adapter is not an instance of TFModelAdapter.

        Returns:
            (AgentBase): An instance of the agent created using the model blueprint and trading environment.
        """

        parameters_needed_for_instantiation = model_blue_print.report_parameters_needed_for_instantiation()
        kwargs = {}
        for parameter in parameters_needed_for_instantiation:
            kwargs[parameter] = self._provide_required_parameter(parameter, trading_environment)
        model_adapter = model_blue_print.instantiate_model(**kwargs)

        if not isinstance(model_adapter, TFModelAdapter):
            raise TypeError("Model adapter must be an instance of TFModelAdapter for reinforcement learning.")

        return ReinforcementLearningAgent(model_adapter.get_model(), self.__policy, self.__optimizer)

    def fit(self, agent: ReinforcementLearningAgent, trading_environment: TradingEnvironment,
            nr_of_steps: int, nr_of_episodes: int, callbacks: list[Callback]) -> tuple[list[str], dict[str, Any]]:
        """
        Fits the reinforcement learning agent to the trading environment.

        Parameters:
            agent (ReinforcementLearningAgent): The reinforcement learning agent to fit.
            trading_environment (TradingEnvironment): The trading environment to use.
            nr_of_steps (int): The number of training steps to perform.
            nr_of_episodes (int): The number of training episodes to perform.
            callbacks (list[Callback]): List of callbacks to use during training.

        Raises:
            TypeError: If the agent is not an instance of ReinforcementLearningAgent.

        Returns:
            (list[str], dict[str, Any]): A tuple containing the keys and data collected during training.
        """

        if not isinstance(agent, ReinforcementLearningAgent):
            raise TypeError("Agent must be an instance of ReinforcementLearningAgent.")

        keys, data = super().fit(agent, trading_environment, nr_of_steps, nr_of_episodes, callbacks)
        keys.append(self.PLOTTING_KEY)

        steps_per_episode = nr_of_steps // nr_of_episodes
        reinforcement_learning_history = agent.reinforcement_learning_fit(trading_environment, nr_of_steps, steps_per_episode, callbacks)
        data.append(reinforcement_learning_history)

        return keys, data

    def _provide_input_shape(self, trading_environment: TradingEnvironment) -> tuple[int, int]:
        """
        Provides the input shape for the model based on the trading environment.

        Parameters:
            trading_environment (TradingEnvironment): The trading environment to use.

        Returns:
            (tuple[int, int]): The input shape for the model.
        """

        return trading_environment.observation_space.shape

    def _provide_output_length(self, trading_environment: TradingEnvironment) -> int:
        """
        Provides the output length for the model based on the trading environment.

        Parameters:
            trading_environment (TradingEnvironment): The trading environment to use.

        Returns:
            (int): The output length for the model.
        """

        return trading_environment.action_space.n

    def _provide_spatial_data_shape(self, trading_environment: TradingEnvironment) -> tuple[int, int]:
        """
        Provides the spatial data shape for the model based on the trading environment.

        Parameters:
            trading_environment (TradingEnvironment): The trading environment to use.

        Returns:
            (tuple[int, int]): The spatial data shape for the model.
        """

        return trading_environment.get_environment_spatial_data_dimension()