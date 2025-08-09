# agent/agents/simple_trading_algorithm.py

# global imports
import numpy as np

# local imports
from source.agent import TradingAlgorithmBase

class SimpleTradingAlgorithm(TradingAlgorithmBase):
    """
    Implements a simple trading algorithm that performs actions based on trend predictions.
    """

    def perform_action(self, trend_predictions: int, market_data: np.ndarray = None) -> int:
        """
        Performs the trading action based on the trend predictions and market data.

        Parameters:
            trend_predictions (np.ndarray): The predicted trends.
            market_data (np.ndarray): The current market data. This
                implementation does not use market data.

        Returns:
            (int): The result of the trading action.
        """

        return trend_predictions