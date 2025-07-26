# training/training_config.py

# global imports
import pandas as pd
from typing import Any, Optional

# local imports
from source.agent import AgentHandler, LearningStrategyHandlerBase, TestingStrategyHandlerBase
from source.environment import LabelAnnotatorBase, LabeledDataBalancer, PriceRewardValidator, \
    RewardValidatorBase, SimpleLabelAnnotator, TradingEnvironment
from source.model import BluePrintBase

class TrainingConfig():
    """
    Implements a configuration class for training agents in a trading environment. It encapsulates
    all necessary parameters for training, including the number of steps, episodes, model blueprint,
    data path, initial budget, maximum amount of trades, window size, and various reward parameters.
    It also provides methods for instantiating an agent handler and printing the configuration.
    """

    def __init__(self, nr_of_steps: int, nr_of_episodes: int, model_blue_print: BluePrintBase,
                 data: pd.DataFrame, initial_budget: float, max_amount_of_trades: int, window_size: int,
                 learning_strategy_handler: LearningStrategyHandlerBase,
                 testing_strategy_handler: TestingStrategyHandlerBase, sell_stop_loss: float = 0.8,
                 sell_take_profit: float = 1.2, buy_stop_loss: float = 0.8, buy_take_profit: float = 1.2,
                 penalty_starts: int = 0, penalty_stops: int = 10, static_reward_adjustment: float = 1,
                 repeat_test: int = 10, test_ratio: float = 0.2, validator: Optional[RewardValidatorBase] = None,
                 label_annotator: Optional[LabelAnnotatorBase] = None,
                 labeled_data_balancer: Optional[LabeledDataBalancer] = None,
                 meta_data: Optional[dict[str, Any]] = None) -> None:
        """
        Class constructor. Initializes the training configuration with the provided parameters.

        Parameters:
            nr_of_steps (int): The number of training steps to perform.
            nr_of_episodes (int): The number of training episodes to perform.
            model_blue_print (BluePrintBase): The blueprint for the model to be trained.
            data (pd.DataFrame): The training data.
            initial_budget (float): The initial budget for the trading agent.
            max_amount_of_trades (int): The maximum number of trades to perform.
            window_size (int): The size of the observation window.
            learning_strategy_handler (LearningStrategyHandlerBase): The handler for the learning strategy.
            testing_strategy_handler (TestingStrategyHandlerBase): The handler for the testing strategy.
            sell_stop_loss (float): The stop loss threshold for selling.
            sell_take_profit (float): The take profit threshold for selling.
            buy_stop_loss (float): The stop loss threshold for buying.
            buy_take_profit (float): The take profit threshold for buying.
            penalty_starts (int): The step at which to start applying penalties.
            penalty_stops (int): The step at which to stop applying penalties.
            static_reward_adjustment (float): The static adjustment factor for rewards.
            repeat_test (int): The number of times to repeat testing.
            test_ratio (float): The ratio of data to use for testing.
            validator (Optional[RewardValidatorBase]): The reward validator to use. Defaults to PriceRewardValidator.
            label_annotator (Optional[LabelAnnotatorBase]): The label annotator to use. Defaults to SimpleLabelAnnotator.
            labeled_data_balancer (Optional[LabeledDataBalancer]): The labeled data balancer to use. Defaults to None.
            meta_data (Optional[dict[str, Any]]): Optional metadata for the training configuration.
        """

        if validator is None:
            validator = PriceRewardValidator()

        if label_annotator is None:
            label_annotator = SimpleLabelAnnotator()

        # Training config
        self.nr_of_steps: int = nr_of_steps
        self.nr_of_episodes: int = nr_of_episodes
        self.repeat_test: int = repeat_test

        # Environment config
        self.__data: pd.DataFrame = data
        self.__meta_data: Optional[dict[str, Any]] = meta_data
        self.__test_ratio: float = test_ratio
        self.__initial_budget: float = initial_budget
        self.__max_amount_of_trades: int = max_amount_of_trades
        self.__window_size: int = window_size
        self.__sell_stop_loss: float = sell_stop_loss
        self.__sell_take_profit: float = sell_take_profit
        self.__buy_stop_loss: float = buy_stop_loss
        self.__buy_take_profit: float = buy_take_profit
        self.__penalty_starts: int = penalty_starts
        self.__penalty_stops: int = penalty_stops
        self.__static_reward_adjustment: float = static_reward_adjustment
        self.__validator: RewardValidatorBase = validator
        self.__label_annotator: LabelAnnotatorBase = label_annotator
        self.__labeled_data_balancer: Optional[LabeledDataBalancer] = labeled_data_balancer

        # Agent config
        self.__model_blue_print: BluePrintBase = model_blue_print
        self.__learning_strategy_handler: LearningStrategyHandlerBase = learning_strategy_handler
        self.__testing_strategy_handler: TestingStrategyHandlerBase = testing_strategy_handler

    def __str__(self) -> str:
        """
        Returns a string representation of the configuration.

        Creates a formatted multi-line string containing all configuration
        parameters and their values for easy logging.

        Returns:
            str: Formatted string representation of the configuration.
        """

        labeled_data_balancer_info = ""
        if self.__labeled_data_balancer is not None:
            labeled_data_balancer_info = \
                f"\tlabeled_data_balancer: {self.__labeled_data_balancer.__class__.__name__}\n" \
                f"\t\t{vars(self.__labeled_data_balancer)}\n"
        else:
            labeled_data_balancer_info = "\tlabeled_data_balancer: None\n"

        return f"Training config:\n" \
                f"\tnr_of_steps: {self.nr_of_steps}\n" \
                f"\tnr_of_episodes: {self.nr_of_episodes}\n" \
                f"\trepeat_test: {self.repeat_test}\n" \
                f"\ttest_ratio: {self.__test_ratio}\n" \
                f"\tinitial_budget: {self.__initial_budget}\n" \
                f"\tmax_amount_of_trades: {self.__max_amount_of_trades}\n" \
                f"\twindow_size: {self.__window_size}\n" \
                f"\tsell_stop_loss: {self.__sell_stop_loss}\n" \
                f"\tsell_take_profit: {self.__sell_take_profit}\n" \
                f"\tbuy_stop_loss: {self.__buy_stop_loss}\n" \
                f"\tbuy_take_profit: {self.__buy_take_profit}\n" \
                f"\tpenalty_starts: {self.__penalty_starts}\n" \
                f"\tpenalty_stops: {self.__penalty_stops}\n" \
                f"\tstatic_reward_adjustment: {self.__static_reward_adjustment}\n" \
                f"\tvalidator: {self.__validator.__class__.__name__}\n" \
                f"\t\t{vars(self.__validator)}\n" \
                f"\tlabel_annotator: {self.__label_annotator.__class__.__name__}\n" \
                f"\t\t{vars(self.__label_annotator)}\n" \
                f"{labeled_data_balancer_info}" \
                f"\tmodel_blue_print: {self.__model_blue_print.__class__.__name__}\n" \
                f"\t\t{vars(self.__model_blue_print)}\n" \
                f"\tlearning_strategy_handler: {self.__learning_strategy_handler.__class__.__name__}\n" \
                f"\t\t{vars(self.__learning_strategy_handler)}\n" \
                f"\ttesting_strategy_handler: {self.__testing_strategy_handler.__class__.__name__}\n" \
                f"\t\t{vars(self.__testing_strategy_handler)}\n"

    def instantiate_agent_handler(self) -> AgentHandler:
        """
        Instantiates the agent handler with the configured environment and strategies.

        Returns:
            (AgentHandler): An instance of the agent handler configured with the model blueprint,
            trading environment, learning strategy handler and testing strategy handler.
        """

        environment = TradingEnvironment(self.__data, self.__initial_budget, self.__max_amount_of_trades,
                                         self.__window_size, self.__validator, self.__label_annotator,
                                         self.__sell_stop_loss, self.__sell_take_profit, self.__buy_stop_loss,
                                         self.__buy_take_profit, self.__test_ratio, self.__penalty_starts,
                                         self.__penalty_stops, self.__static_reward_adjustment,
                                         self.__labeled_data_balancer, self.__meta_data)

        return AgentHandler(self.__model_blue_print, environment, self.__learning_strategy_handler,
                            self.__testing_strategy_handler)