# agent/strategies/classification_testable.py

# global imports
import numpy as np
from abc import ABC, abstractmethod

# local imports

class ClassificationTestable(ABC):
    """
    Implements an abstract base class for classification testable agents.
    """

    @abstractmethod
    def classify(self, data: np.ndarray) -> list[list[float]]:
        """
        Classifies the input data using the trained model.

        Parameters:
            data (np.ndarray): The input data to be classified.

        Returns:
            (list[list[float]]): The predicted class probabilities for each input sample.
        """

        pass