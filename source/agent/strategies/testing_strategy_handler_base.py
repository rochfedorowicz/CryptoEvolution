# agent/strategies/testing_strategy_handler_base.py

# global imports
from abc import ABC, abstractmethod
from typing import Any

# local imports
from source.agent import AgentBase
from source.environment import TradingEnvironment

class TestingStrategyHandlerBase(ABC):
    """
    Implements a base class for testing strategy handlers. It expects derived classes to implement
    functions for evaluating agents in a trading environment.
    """

    @abstractmethod
    def evaluate(self, agent: AgentBase, environment: TradingEnvironment,
        env_length_range: tuple[int, int]) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Evaluates the performance of the given agent in the specified trading environment.

        Parameters:
            agent (AgentBase): The agent to evaluate.
            environment (TradingEnvironment): The trading environment to use for evaluation.
            env_length_range (tuple[int, int]): A tuple specifying the range of environment lengths to consider.

        Returns:
            (tuple[list[str], list[dict[str, Any]]]): A tuple containing the keys and data collected during evaluation.
        """

        pass