# agent/agents/reinforcement_learning_agent.py

# global imports
import rl
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import Policy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from typing import Any, Callable, Optional

# local imports
from source.agent import AgentBase, PerformanceTestable
from source.environment import TradingEnvironment

class ReinforcementLearningAgent(AgentBase, PerformanceTestable):
    """
    Implements a reinforcement learning agent using DQN that can be trained and tested
    in a trading environment. It provides functionalities for fitting the model
    with reinforcement learning data and performing actions based on observations.
    """

    def __init__(self, model: Model, policy: Policy, optimizer: Optimizer) -> None:
        """
        Class constructor. Creates a reinforcement learning agent with the given model,
        policy, and optimizer.

        Parameters:
            model (Model): The Keras model to use for the agent.
            policy (Policy): The policy to use for the agent's actions.
            optimizer (Optimizer): The optimizer to use for training the agent.
        """

        memory = SequentialMemory(limit = 500000, window_length = 1)
        self.__DQNAgent: DQNAgent = rl.agents.DQNAgent(model, policy, memory = memory,
                                                       nb_actions = model.output_shape[-1],
                                                       target_model_update = 1e-2)
        self.__DQNAgent.compile(optimizer)
        self.__DQNAgent.optimizer = self.__DQNAgent.model.optimizer  # For compatibility with callbacks

    def load_model(self, model_path: str) -> None:
        """
        Loads the model weights from the specified file path.

        Parameters:
            model_path (str): The path to the model weights file.
        """

        self.__DQNAgent.load_weights(model_path)

    def save_model(self, model_path: str) -> None:
        """
        Saves the model weights to the specified file path.

        Parameters:
            model_path (str): The path to the model weights file.
        """

        self.__DQNAgent.save_weights(model_path)

    def print_summary(self, print_function: Optional[Callable] = print) -> None:
        """
        Prints a summary of the model architecture.

        Parameters:
            print_function (Optional[Callable]): A function to use for printing the summary.
                Defaults to the print function.
        """

        self.__DQNAgent.model.summary(print_fn = print_function)

    def reinforcement_learning_fit(self, environment: TradingEnvironment, nr_of_steps: int,
                                   steps_per_episode: int, callbacks: list[Callback]) -> dict[str, Any]:
        """
        Trains the reinforcement learning agent using the specified environment and parameters.

        Parameters:
            environment (TradingEnvironment): The trading environment to use for training.
            nr_of_steps (int): The total number of steps to train the agent.
            steps_per_episode (int): The number of steps per episode.
            callbacks (list[Callback]): A list of Keras callbacks to use during training.

        Returns:
            (dict[str, Any]): The training history of the agent.
        """

        return self.__DQNAgent.fit(environment, nr_of_steps, callbacks = callbacks,
                                   log_interval = steps_per_episode,
                                   nb_max_episode_steps = steps_per_episode).history

    def perform(self, observation: list[float]) -> int:
        """
        Performs the action for the agent based on the given observation.

        Parameters:
            observation (list[float]): The observation data to use for the action.

        Returns:
            (int): The result of the action.
        """

        return self.__DQNAgent.forward(observation)