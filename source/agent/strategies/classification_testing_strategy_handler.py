# agent/strategies/classification_testing_strategy_handler.py

# global imports
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import Any

# local imports
from source.agent import ClassificationTestable, TestingStrategyHandlerBase
from source.environment import TradingEnvironment

class ClassificationTestingStrategyHandler(TestingStrategyHandlerBase):
    """
    Implements a testing strategy handler for classification tasks.
    """

    # global class constants
    PLOTTING_KEY: str = 'classification_testing'

    def evaluate(self, testable_agent: ClassificationTestable, environment: TradingEnvironment,
        env_length_range: tuple[int, int]) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Evaluates the classification model using the given testable agent and trading environment.

        Parameters:
            testable_agent (ClassificationTestable): The agent to be tested.
            environment (TradingEnvironment): The trading environment containing the test data.
            env_length_range (tuple[int, int]): A tuple specifying the range of environment lengths to consider.

        Returns:
            (tuple[list[str], list[dict[str, Any]]]): A tuple containing the keys and data collected during evaluation.
        """

        classes = list(environment.get_trading_consts().OUTPUT_CLASSES.keys())
        input_data, output_data, _, _ = environment.get_labeled_data(env_length_range = env_length_range)
        prediction_probabilities = testable_agent.classify(input_data)
        y_pred = np.argmax(prediction_probabilities, axis = 1)

        conf_matrix = confusion_matrix(output_data, y_pred)
        class_report = classification_report(output_data, y_pred, target_names = classes,
                                             output_dict = True, zero_division = 0)

        summary = {
            "true_labels": output_data,
            "prediction_probabilities": prediction_probabilities,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report
        }

        return [self.PLOTTING_KEY], [summary]