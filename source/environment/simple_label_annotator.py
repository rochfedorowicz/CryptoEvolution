# environment/simple_label_annotator.py

# global imports

# local imports
from source.environment import LabelAnnotatorBase

class SimpleLabelAnnotator(LabelAnnotatorBase):
    """
    Implements a simple label annotator that classifies price movements into three classes:
    - Up trend
    - Down trend
    - No trend
    """

    def __init__(self, threshold: float = 0.01) -> None:
        """
        Class constructor. Initializes the SimpleLabelAnnotator with a specified threshold for trend classification.

        Parameters:
            threshold (float): The threshold for classifying price movements.
        """

        super().__init__()
        self._output_classes.UP_TREND = 0
        self._output_classes.DOWN_TREND = 1
        self._output_classes.NO_TREND = 2
        self.__threshold = threshold

    def _classify_trend(self, price_diff: float) -> int:
        """
        Classifies the price movement trend based on the price difference.

        Parameters:
            price_diff (float): The price difference to classify.

        Returns:
            (int): The class label for the price movement trend.
        """

        if price_diff > self.__threshold:
                return self._output_classes.UP_TREND
        elif price_diff < -self.__threshold:
            return self._output_classes.DOWN_TREND
        else:
            return self._output_classes.NO_TREND
