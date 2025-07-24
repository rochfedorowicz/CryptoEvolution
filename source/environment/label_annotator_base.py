# environment/label_annotator_base.py

# global imports
import pandas as pd
from abc import ABC, abstractmethod
from types import SimpleNamespace

# local imports

class LabelAnnotatorBase(ABC):
    """
    Implements a base class for label annotators. It provides an interface for annotating
    data with labels based on price movements.
    """

    # local constants
    __CLOSE_PRICE_COLUMN_NAME: str = "close"

    @abstractmethod
    def __init__(self) -> None:
        """
        Class constructor. Initializes the output classes for classification.
        """

        self._output_classes = SimpleNamespace()

    @abstractmethod
    def _classify_trend(self, price_diff: float) -> int:
        """
        Classifies the price movement trend based on the price difference.

        Parameters:
            price_diff (float): The price difference to classify.

        Returns:
            (int): The class label for the price movement trend.
        """

        pass

    def annotate(self, data: pd.DataFrame) -> pd.Series:
        """
        Annotates the provided data with labels based on price movements.

        Parameters:
            data (pd.DataFrame): The data to annotate, must contain a 'close' column.

        Returns:
            (pd.Series): A series of labels corresponding to the price movement trends.
        """

        current_prices = data[self.__CLOSE_PRICE_COLUMN_NAME]
        next_day_prices = data[self.__CLOSE_PRICE_COLUMN_NAME].shift(-1)
        price_diffs = (next_day_prices - current_prices) / current_prices

        return price_diffs.dropna().apply(self._classify_trend)

    def get_output_classes(self) -> SimpleNamespace:
        """
        Returns the output classes for the label annotator.

        Returns:
            (SimpleNamespace): The SimpleNamespace containing the labels for classification.
        """

        return self._output_classes