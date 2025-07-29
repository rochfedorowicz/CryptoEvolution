# environment/label_annotator_base.py

# global imports
import pandas as pd
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Optional

# local imports

class LabelAnnotatorBase(ABC):
    """
    Implements a base class for label annotators. It provides an interface for annotating
    data with labels based on price movements.
    """

    # derived constants
    _CLOSE_PRICE_COLUMN_NAME: str = "close"
    _CLOSE_PRICE_CHANGE_COLUMN_NAME: str = "future_normalized_diff"

    def __init__(self) -> None:
        """
        Class constructor. Initializes the output classes for classification.
        """

        self._output_classes: Optional[SimpleNamespace] = None
        self._requested_columns: Optional[list[str]] = None

    @abstractmethod
    def _classify_trend(self, row: pd.Series) -> int:
        """
        Classifies the price movement trend based on the price difference.

        Parameters:
            row (pd.Series): The row of with data requested to classify.

        Returns:
            (int): The class label for the price movement trend.
        """

        pass

    def annotate(self, data: pd.DataFrame) -> pd.Series:
        """
        Annotates the provided data with labels based on price movements.

        Parameters:
            data (pd.DataFrame): The data to annotate, must contain a 'close' column.

        Raises:
            ValueError: If the output classes are not initialized before annotating data.

        Returns:
            (pd.Series): A series of labels corresponding to the price movement trends.
        """

        if self._output_classes is None:
            raise ValueError("Output classes must be initialized in derived classes " \
                "before annotating data.")

        if self._requested_columns is None:
            self._requested_columns = [self._CLOSE_PRICE_CHANGE_COLUMN_NAME]

        if self._CLOSE_PRICE_CHANGE_COLUMN_NAME in self._requested_columns:
            current_prices = data[self._CLOSE_PRICE_COLUMN_NAME]
            next_day_prices = data[self._CLOSE_PRICE_COLUMN_NAME].shift(-1)
            future_normalized_diff = (next_day_prices - current_prices) / current_prices
            data[self._CLOSE_PRICE_CHANGE_COLUMN_NAME] = future_normalized_diff

        if missing_columns := set(self._requested_columns) - set(data.columns):
            raise ValueError(f"Data is missing required columns: {missing_columns}")

        return data[self._requested_columns].apply(self._classify_trend, axis = 1)[:-1]

    def get_output_classes(self) -> SimpleNamespace:
        """
        Returns the output classes for the label annotator.

        Returns:
            (SimpleNamespace): The SimpleNamespace containing the labels for classification.
        """

        return self._output_classes