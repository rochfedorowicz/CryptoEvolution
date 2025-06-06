# indicators/indicator_base.py

# global imports
import pandas as pd
from abc import ABC, abstractmethod

# local imports

class IndicatorHandlerBase(ABC):
    """
    Base class for indicators. Enforces certain functions to be implemented
    in derivative classes.
    """

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated values for certain indicator.
        """

        pass
