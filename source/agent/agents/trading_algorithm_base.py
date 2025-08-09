# agent/agents/trading_algorithm_base.py

# global imports
import numpy as np
from abc import ABC, abstractmethod

# local imports

class TradingAlgorithmBase(ABC):
    """
    Implements a base class for trading algorithms. It defines the perform_action method
    that must be implemented by subclasses.
    """

    @abstractmethod
    def perform_action(self, trend_predictions: int, market_data: np.ndarray) -> int:
        """
        Performs the trading action based on the trend predictions and market data.

        Parameters:
            trend_predictions (np.ndarray): The predicted trends.
            market_data (np.ndarray): The current market data.

        Returns:
            (int): The result of the trading action.
        """

        pass