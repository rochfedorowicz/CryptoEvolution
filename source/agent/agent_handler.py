# agent/agent_handler_base.py

# global imports
import logging
import random
from tensorflow.keras.callbacks import Callback
from typing import Any, Callable, Optional

# local imports
from source.agent import AgentBase, LearningStrategyHandlerBase, TestingStrategyHandlerBase
from source.environment import TradingEnvironment
from source.model import BluePrintBase
from source.utils import redirect_stdout_to_logging

class AgentHandler():
    """
    Implements agent handler that is responsible for training and testing
    the agent in the given trading environment using the specified learning
    and testing strategies. It is used as an wrapper around the agent
    to provide a convenient interface for TrainingHandler.
    """

    def __init__(self, model_blue_print: BluePrintBase,
                 trading_environment: TradingEnvironment,
                 learning_strategy_handler: LearningStrategyHandlerBase,
                 testing_strategy_handler: TestingStrategyHandlerBase) -> None:
        """
        Class constructor. Initializes the agent handler with the given model blueprint,
        trading environment, learning strategy handler, and testing strategy handler.

        Parameters:
            model_blue_print (BluePrintBase): The model blueprint to be used for the agent.
            trading_environment (TradingEnvironment): The trading environment in which the agent will operate.
            learning_strategy_handler (LearningStrategyHandlerBase): The learning strategy handler to be used for training.
            testing_strategy_handler (TestingStrategyHandlerBase): The testing strategy handler to be used for evaluation.
        """

        self.__trained: bool = False
        self.__learning_strategy_handler: LearningStrategyHandlerBase = learning_strategy_handler
        self.__testing_strategy_handler: TestingStrategyHandlerBase = testing_strategy_handler
        self.__trading_environment: TradingEnvironment = trading_environment
        self.__agent: AgentBase = learning_strategy_handler.create_agent(model_blue_print, trading_environment)

    def train_agent(self, nr_of_steps: int, nr_of_episodes: int, callbacks: Optional[list[Callback]] = None,
                    model_load_path: Optional[str] = None,
                    model_save_path: Optional[str] = None) -> tuple[list[str], list[dict]]:
        """
        Trains the agent using the specified number of steps and episodes.

        Parameters:
            nr_of_steps (int): The number of steps to train the agent.
            nr_of_episodes (int): The number of episodes to train the agent.
            callbacks (Optional[list[Callback]]): A list of callbacks to be used during training.
            model_load_path (Optional[str]): Path to load the pre-trained model from.
            model_save_path (Optional[str]): Path to save the trained model to.

        Returns:
            (tuple[list[str], list[dict]]): A tuple containing the keys and report data from
                the training process.
        """

        if callbacks is None:
            callbacks = []

        self.__trading_environment.set_mode(TradingEnvironment.TRAIN_MODE)

        if model_load_path is not None:
            self.__agent.load_model(model_load_path)

        with redirect_stdout_to_logging():
            keys, report_data = self.__learning_strategy_handler.fit(
                self.__agent,
                self.__trading_environment,
                nr_of_steps,
                nr_of_episodes,
                callbacks
            )

        self.__trained = True
        if model_save_path is not None:
            self.__agent.save_model(model_save_path)

        return keys, report_data

    def test_agent(self, repeat: int = 1) -> tuple[dict[int, list[str]], dict[int, list[dict[str, Any]]]]:
        """
        Tests the agent using the specified number of repetitions.

        Parameters:
            repeat (int): The number of times to repeat the testing process.

        Returns:
            (tuple[dict[int, list[str]], dict[int, list[dict[str, Any]]]]): A tuple containing the keys and
                report data from the testing process.
        """

        if not self.__trained:
            logging.error('Agent is not trained yet! Train the agent before testing.')
            return {}, {}

        self.__trading_environment.set_mode(TradingEnvironment.TEST_MODE)

        report_data = {}
        keys = {}
        for i in range(repeat):
            env_length = self.__trading_environment.get_environment_length()
            window_size = self.__trading_environment.get_trading_consts().WINDOW_SIZE
            current_iteration = random.randint(window_size, int(env_length/2))
            self.__trading_environment.reset(current_iteration)
            keys[i], report_data[i] = self.__testing_strategy_handler.evaluate(self.__agent,
                                                                               self.__trading_environment)

        return keys, report_data

    def print_model_summary(self, print_function: Optional[Callable] = print) -> None:
        """
        Prints a summary of the model architecture and parameters.

        Parameters:
            print_function (Optional[Callable]): A function to print the summary. Defaults to print.
        """

        self.__agent.print_summary(print_function = print_function)