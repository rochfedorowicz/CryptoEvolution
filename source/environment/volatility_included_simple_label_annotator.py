# environment/volatility_included_simple_label_annotator.py

# global imports
import pandas as pd
from types import SimpleNamespace

# local imports
from source.environment import LabelAnnotatorBase

class VolatilityIncludedSimpleLabelAnnotator(LabelAnnotatorBase):
    """
    Implements a simple label annotator that classifies price movements into three classes:
    - Up trend
    - Down trend
    - No trend
    This annotator also includes volatility in the classification process.
    """

    # local constants
    __VOLATILITY_COLUMN_NAME: str = "volatility"

    def __init__(self, threshold: float = 0.1) -> None:
        """
        Class constructor. Initializes the SimpleLabelAnnotator with a specified threshold for trend classification.

        Parameters:
            threshold (float): The threshold for classifying price movements.
        """

        super().__init__()
        self._output_classes = SimpleNamespace()
        self._output_classes.UP_TREND = 0
        self._output_classes.NO_TREND = 1
        self._output_classes.DOWN_TREND = 2
        self.__threshold = threshold
        self._requested_columns = [self._CLOSE_PRICE_CHANGE_COLUMN_NAME,
                                   self.__VOLATILITY_COLUMN_NAME]

    def _classify_trend(self, row: pd.Series) -> int:
        """
        Classifies the price movement trend based on the price difference.

        Parameters:
            row (pd.Series): The row of with data requested to classify.

        Returns:
            (int): The class label for the price movement trend.
        """

        price_diff = row[self._CLOSE_PRICE_CHANGE_COLUMN_NAME]
        volatility = row[self.__VOLATILITY_COLUMN_NAME]

        if price_diff > self.__threshold * volatility:
                return self._output_classes.UP_TREND
        elif price_diff < -self.__threshold * volatility:
            return self._output_classes.DOWN_TREND
        else:
            return self._output_classes.NO_TREND
