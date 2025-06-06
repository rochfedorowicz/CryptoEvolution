# agent/strategies/performance_testable.py

# global imports
from abc import ABC, abstractmethod

# local imports

class PerformanceTestable(ABC):
    """
    Implements a performance testable interface for agents.
    """

    @abstractmethod
    def perform(self, observation: list[float]) -> int:
        """
        Performs the action for the agent based on the given observation.

        Parameters:
            observation (list[float]): The observation data to use for the action.

        Returns:
            (int): The result of the action.
        """

        pass