# agent/agents/performance_testable_classification_learning_agent.py

# global imports
import numpy as np
from typing import Optional

# local imports
from source.agent import ClassificationLearningAgent, PerformanceTestable, SimpleTradingAlgorithm, TradingAlgorithmBase
from source.model import ModelAdapterBase

class PerformanceTestableClassificationLearningAgent(ClassificationLearningAgent, PerformanceTestable):
    """
    Class that extends ClassificationLearningAgent to include performance testing capabilities.
    """

    def __init__(self, model_adapter: ModelAdapterBase, trading_algorithm: Optional[TradingAlgorithmBase] = None) -> None:
        """
        Class constructor. Initializes the agent with a model adapter and an optional trading algorithm.

        Parameters:
            model_adapter (ModelAdapterBase): The model adapter to use for the agent.
            trading_algorithm (Optional[TradingAlgorithmBase]): The trading algorithm to use for the agent.
                Defaults to SimpleTradingAlgorithm.
        """

        if trading_algorithm is None:
            trading_algorithm = SimpleTradingAlgorithm()

        super().__init__(model_adapter)
        self.__trading_algorithm: TradingAlgorithmBase = trading_algorithm

    def perform(self, observation: list[float]) -> int:
        """
        Performs the action for the agent based on the given observation.

        Parameters:
            observation (list[float]): The observation data to use for the action.

        Returns:
            (int): The result of the action.
        """

        trend_predictions = np.argmax(super().classify(np.array(observation[:-3]).reshape(1, -1))[0])
        return self.__trading_algorithm.perform_action(trend_predictions, observation[-3:])
