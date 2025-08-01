# environment/trading environment.py

# global imports
import copy
import logging
import math
import numpy as np
import pandas as pd
import random
from enum import Enum
from gym import Env
from gym.spaces import Box, Discrete
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from types import SimpleNamespace
from typing import Any, Optional

# local imports
from source.environment import Broker, LabelAnnotatorBase, LabeledDataBalancer, RewardValidatorBase

class TradingEnvironment(Env):
    """
    Implements stock market environment that actor can perform actions (place orders) in.
    It is used to train various models using various approaches. Can be
    configured to award points and impose a penalty in several ways.
    """

    class TradingMode(Enum):
        """
        Enumeration for the different trading modes.
        """

        IMPLICIT_ORDER_CLOSING = 0
        EXPLICIT_ORDER_CLOSING = 1

    # global class constants
    TRAIN_MODE = 'train'
    TEST_MODE = 'test'

    def __init__(self, data: pd.DataFrame, initial_budget: float, max_amount_of_trades: int,
                 window_size: int, validator: RewardValidatorBase, label_annotator: LabelAnnotatorBase,
                 sell_stop_loss: float, sell_take_profit: float, buy_stop_loss: float, buy_take_profit: float,
                 test_ratio: float = 0.2, penalty_starts: int = 0, penalty_stops: int = 10,
                 static_reward_adjustment: float = 1, labeled_data_balancer: Optional[LabeledDataBalancer] = None,
                 meta_data: Optional[dict[str, Any]] = None, trading_mode: Optional[TradingMode] = None,
                 should_prefetch: bool = True) -> None:
        """
        Class constructor. Allows to define all crucial constans, reward validation methods,
        environmental penalty policies, etc.

        Parameters:
            data (pd.DataFrame): DataFrame containing historical market data.
            initial_budget (float): Initial budget constant for trader to start from.
            max_amount_of_trades (int): Max amount of trades that can be ongoing at the same time.
                Seting this constant prevents traders from placing orders randomly and defines
                amount of money that can be assigned to a single order at certain iteration.
            window_size (int): Constant defining how far in the past trader will be able to look
                into at certain iteration.
            validator (RewardValidatorBase): Validator implementing policy used to award points
                for closed trades.
            label_annotator (LabelAnnotatorBase): Annotator implementing policy used to label
                data with target values. It is used to provide supervised agents with information
                about what is the target class value for certain iteration.
            sell_stop_loss (float): Constant used to define losing boundary at which sell order
                (short) is closed.
            sell_take_profit (float): Constant used to define winning boundary at which sell order
                (short) is closed.
            buy_stop_loss (float): Constant used to define losing boundary at which buy order
                (long) is closed.
            buy_take_profit (float): Constant used to define winning boundary at which buy order
                (long) is closed.
            test_ratio (float): Ratio of data that should be used for testing purposes.
            penalty_starts (int): Constant defining how many trading periods can trader go without placing
                an order until penalty is imposed. Penalty at range between start and stop constant
                is calculated as percentile of positive reward, and subtracted from the actual reward.
            penalty_stops (int): Constant defining at which trading period penalty will no longer be increased.
                Reward for trading periods exceeding penalty stop constant will equal minus static reward adjustment.
            static_reward_adjustment (float): Constant use to penalize trader for bad choices or
                reward it for good one.
            labeled_data_balancer (Optional[LabeledDataBalancer]): Balancer used to balance
                labeled data. If None, no balancing will be performed.
            meta_data (dict[str, Any]): Dictionary containing metadata about the dataset.
            mode (TradingMode): Mode of the environment, either IMPLICIT_ORDER_CLOSING or EXPLICIT_ORDER_CLOSING.
            should_prefetch (bool): If True, data will be pre-fetched to speed up training.
        """

        if test_ratio < 0.0 or test_ratio >= 1.0:
            raise ValueError(f"Invalid test_ratio: {test_ratio}. It should be in range [0, 1).")

        if trading_mode is None:
            trading_mode = TradingEnvironment.TradingMode.IMPLICIT_ORDER_CLOSING

        # Initializing the environment
        self.__data: dict[pd.DataFrame, pd.DataFrame] = self.__split_data(data, test_ratio)
        self.__meta_data: Optional[dict[str, Any]] = meta_data
        self.__mode = TradingEnvironment.TRAIN_MODE
        self.__trading_mode: TradingEnvironment.TradingMode = trading_mode
        self.__should_prefetch: bool = should_prefetch
        self.__broker: Broker = Broker()
        self.__validator: RewardValidatorBase = validator
        self.__label_annotator: LabelAnnotatorBase = label_annotator
        self.__labeled_data_balancer: Optional[LabeledDataBalancer] = labeled_data_balancer

        # Setting up trading data
        self.__trading_data: SimpleNamespace = SimpleNamespace()
        self.__trading_data.current_budget: float = initial_budget
        self.__trading_data.currently_invested: float = 0
        self.__trading_data.no_trades_placed_for: int = 0
        self.__trading_data.currently_placed_trades: int = 0

        # Setting up trading constants
        self.__trading_consts = SimpleNamespace()
        self.__trading_consts.INITIAL_BUDGET: float = initial_budget
        self.__trading_consts.MAX_AMOUNT_OF_TRADES: int = max_amount_of_trades
        self.__trading_consts.WINDOW_SIZE: int = window_size
        self.__trading_consts.SELL_STOP_LOSS: float = sell_stop_loss
        self.__trading_consts.SELL_TAKE_PROFIT: float = sell_take_profit
        self.__trading_consts.BUY_STOP_LOSS: float = buy_stop_loss
        self.__trading_consts.BUY_TAKE_PROFIT: float = buy_take_profit
        self.__trading_consts.STATIC_REWARD_ADJUSTMENT: float = static_reward_adjustment
        self.__trading_consts.PENALTY_STARTS: int = penalty_starts
        self.__trading_consts.PENALTY_STOPS: int = penalty_stops
        self.__trading_consts.PROFITABILITY_FUNCTION = lambda x: -1.0 * math.exp(-x + 1) + 1
        self.__trading_consts.PENALTY_FUNCTION = lambda x: \
            min(1, 1 - math.tanh(-3.0 * (x - penalty_stops) / (penalty_stops - penalty_starts)))
        self.__trading_consts.OUTPUT_CLASSES: int = vars(self.__label_annotator.get_output_classes())

        # Prefetching data if needed
        if self.__should_prefetch:
            self.__prefetched_data = { TradingEnvironment.TRAIN_MODE: None,
                                    TradingEnvironment.TEST_MODE: None }
            self.__mode = TradingEnvironment.TEST_MODE
            self.__prefetched_data[self.__mode] = self.__prefetch_state_data(env_length_range = (self.__trading_consts.WINDOW_SIZE,
                                                                                                 self.get_environment_length()))
            self.__mode = TradingEnvironment.TRAIN_MODE
            self.__prefetched_data[self.__mode] = self.__prefetch_state_data(env_length_range = (self.__trading_consts.WINDOW_SIZE,
                                                                                                 self.get_environment_length()))
        else:
            self.__prefetched_data = None

        # Initializing the environment state
        self.current_iteration: int = self.__trading_consts.WINDOW_SIZE
        self.state: list[float] = self.__get_current_state_data()
        self.action_space: Discrete = Discrete(3)
        self.observation_space: Box = Box(low = np.ones(len(self.state)) * -3,
                                          high = np.ones(len(self.state)) * 3,
                                          dtype = np.float64)

    def __split_data(self, data: pd.DataFrame, test_size: float) -> dict[pd.DataFrame, pd.DataFrame]:
        """
        Splits the given DataFrame into training and testing sets based on the specified test size ratio.

        Parameters:
            data (pd.DataFrame): DataFrame containing the stock market data.
            test_size (float): Ratio of the data to be used for testing.

        Returns:
            (dict[pd.DataFrame, pd.DataFrame]): Dictionary containing training and testing data frames.
        """

        dividing_index = int(len(data) * (1 - test_size))

        return {
            TradingEnvironment.TRAIN_MODE: data.iloc[:dividing_index].reset_index(drop=True),
            TradingEnvironment.TEST_MODE: data.iloc[dividing_index:].reset_index(drop=True)
        }

    def __standard_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Standardizes the given data by removing the mean and scaling to unit variance.

        Parameters:
            data (np.ndarray): The data to be standardized.

        Returns:
            (np.ndarray): The standardized data.
        """

        mean = np.mean(data, axis = 0, keepdims = True)
        std = np.std(data, axis = 0, keepdims = True)
        std[std == 0] = 1

        return (data - mean) / std

    def __prepare_state_data(self, slice_to_get: slice, include_trading_data: bool = True) -> list[float]:
        """
        Calculates state data as a list of floats representing current iteration's observation.
        Observations contains all input data refined to window size and couple of coefficients
        giving an insight into current budget and orders situation.

        Parameters:
            slice (slice): Slice to get the data from.
            include_trading_data (bool): If True, includes trading data in the observation.

        Returns:
           (list[float]): List with current observations for environment.
        """

        current_market_data = self.__data[self.__mode].iloc[slice_to_get].copy()
        current_market_data_no_index = current_market_data.select_dtypes(include = [np.number])

        if self.__meta_data is not None and \
            self.__meta_data.get('normalization_groups', None) is not None:
            grouped_columns_names = self.__meta_data['normalization_groups']
            preprocessed_data_pieces = []
            left_over_columns_names = set(current_market_data_no_index.columns)
            for columns_names_to_normalize in grouped_columns_names:
                left_over_columns_names -= set(columns_names_to_normalize)
                data_frame_piece_to_normalize = current_market_data_no_index[columns_names_to_normalize]
                normalized_data_frame_piece = self.__standard_scale(data_frame_piece_to_normalize.values.reshape(-1, 1))
                preprocessed_data_pieces.append(normalized_data_frame_piece.reshape(*data_frame_piece_to_normalize.shape))
            for column in left_over_columns_names:
                preprocessed_data_pieces.append(current_market_data_no_index[column].values.reshape(-1, 1))
            normalized_current_market_data_values = np.hstack(preprocessed_data_pieces)
        else:
            normalized_current_market_data_values = self.__standard_scale(current_market_data_no_index.values)
        current_marked_data_list = normalized_current_market_data_values.ravel().tolist()

        if include_trading_data:
            current_normalized_budget = 1.0 * self.__trading_data.current_budget / self.__trading_consts.INITIAL_BUDGET
            current_profitability_coeff = self.__trading_consts.PROFITABILITY_FUNCTION(current_normalized_budget)
            current_trades_occupancy_coeff = 1.0 * self.__trading_data.currently_placed_trades  / self.__trading_consts.MAX_AMOUNT_OF_TRADES
            current_no_trades_penalty_coeff = self.__trading_consts.PENALTY_FUNCTION(self.__trading_data.no_trades_placed_for)
            current_inner_state_list = [current_profitability_coeff, current_trades_occupancy_coeff, current_no_trades_penalty_coeff]
            current_marked_data_list += current_inner_state_list

        return current_marked_data_list

    def __prefetch_state_data(self, env_length_range: tuple[int, int], include_trading_data: bool = True) -> pd.DataFrame:
        """
        Prefetches state data for the given environment length range.

        Parameters:
            env_length_range (tuple[int, int]): Range to limit the length of the environment.
            include_trading_data (bool): If True, includes trading data in the observation.

        Returns:
            (pd.DataFrame): DataFrame containing the pre-fetched state data.
        """

        new_rows = []
        for i in range(env_length_range[0], env_length_range[1]):
            data_row = self.__prepare_state_data(slice(i - self.__trading_consts.WINDOW_SIZE, i), include_trading_data = include_trading_data)
            new_rows.append(data_row)

        return pd.DataFrame(new_rows, columns = [f"feature_{i}" for i in range(len(new_rows[0]))])

    def __prepare_labeled_data(self, env_length_range: tuple[int, int]) -> pd.DataFrame:
        """
        Prepares labeled data for training the model with classification approach.
        It extracts the relevant features and labels from the environment's data.

        Parameters:
            env_length_range (tuple[int, int]): Range to limit the length

        Returns:
            (pd.DataFrame): A DataFrame containing the features and labels for training.
        """

        prefetched_data = self.__prefetch_state_data(env_length_range, include_trading_data = False)
        labels = self.__label_annotator.annotate(self.__data[self.__mode]. \
            iloc[:env_length_range[1]].copy()).shift(-env_length_range[0]).dropna()

        return prefetched_data, labels

    def __get_current_state_data(self) -> list[float]:
        """
        Retrieves the current state data from the environment.

        Returns:
            (list[float]): List with current observations for environment.
        """

        if self.__should_prefetch:
            return self.__prefetched_data[self.__mode].iloc[self.current_iteration - self.__trading_consts.WINDOW_SIZE].values.ravel().tolist()

        return self.__prepare_state_data(slice_to_get = slice(self.current_iteration - self.__trading_consts.WINDOW_SIZE, self.current_iteration))


    def set_mode(self, mode: str) -> None:
        """
        Sets the mode of the environment to either TRAIN_MODE or TEST_MODE.

        Parameters:
            mode (str): Mode to set for the environment.

        Raises:
            ValueError: If the provided mode is not valid.
        """

        if mode not in [TradingEnvironment.TRAIN_MODE, TradingEnvironment.TEST_MODE]:
            raise ValueError(f"Invalid mode: {mode}. Use TradingEnvironment.TRAIN_MODE or TradingEnvironment.TEST_MODE.")
        self.__mode = mode

    def get_mode(self) -> str:
        """
        Mode getter.

        Returns:
            (str): Current mode of the environment.
        """

        return copy.copy(self.__mode)

    def get_trading_data(self) -> SimpleNamespace:
        """
        Trading data getter.

        Returns:
            (SimpleNamespace): Copy of the namespace with all trading data.
        """

        return copy.copy(self.__trading_data)

    def get_number_of_trading_points_per_year(self) -> int:
        """
        Returns the number of trading points per year.

        Returns:
            (int): Number of trading points per year.
        """

        temp_data = {"time": pd.to_datetime(self.__data[self.TRAIN_MODE]['time'])}
        temp_df = pd.DataFrame(temp_data)
        temp_df['year'] = temp_df['time'].dt.year

        trading_points_per_year = temp_df.groupby('year').size()
        if len(trading_points_per_year) > 3:
            # If there are more than three years, return the mode
            # of the central years
            return trading_points_per_year.iloc[1:-1].mode()[0]
        elif len(trading_points_per_year) > 2:
            # If there are only three years, return the middle year
            return trading_points_per_year.values[-2]
        else:
            # If there are only two years, return the maximum
            return max(trading_points_per_year.values)

    def get_trading_consts(self) -> SimpleNamespace:
        """
        Trading constants getter.

        Returns:
            (SimpleNamespace): Copy of the namespace with all trading constants.
        """

        return copy.copy(self.__trading_consts)

    def get_broker(self) -> Broker:
        """
        Broker getter.

        Returns:
            (Broker): Copy of the broker used by environment.
        """

        return copy.copy(self.__broker)

    def get_environment_length(self) -> int:
        """
        Environment length getter.

        Returns:
            (Int): Length of environment.
        """

        return len(self.__data[self.__mode])

    def get_environment_spatial_data_dimension(self) -> tuple[int, int]:
        """
        Environment spatial data dimensionality getter.

        Returns:
            (Int): Dimension of spatial data in environment.
        """

        return (self.__trading_consts.WINDOW_SIZE, self.__data[self.__mode].shape[1] - 1)

    def get_labeled_data(self, should_split: bool = True, should_balance: bool = True,
        verbose: bool = True, env_length_range: Optional[tuple[int, int]] = None) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares labeled data for training or testing the model.
        It extracts the relevant features and labels from the environment's data.

        Parameters:
            should_split (bool): Whether to split the data into training and testing sets.
                Defaults to True. If set to False, testing data will be empty.
            should_balance (bool): Whether to balance the labeled data. Defaults to True.
                Will be ignored if labeled_data_balancer is None.
            verbose (bool): Whether to log the class cardinality before and after balancing.
                Defaults to True.
            env_length_range (tuple[int, int]): Optional range to limit the range of the environment.

        Returns:
            (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple containing the
                input data, output data, test input data, and test output data.
        """

        if env_length_range is None:
            env_length_range = (self.__trading_consts.WINDOW_SIZE, self.get_environment_length() - 1)

        input_data, output_data = self.__prepare_labeled_data(env_length_range)
        input_data_test, output_data_test = [], []
        if verbose:
            logging.info(f"Original class cardinality: {np.array(to_categorical(output_data)).sum(axis = 0)}")

        if self.__mode == TradingEnvironment.TRAIN_MODE:
            if should_split:
                input_data, input_data_test, output_data, output_data_test = \
                    train_test_split(input_data, output_data, test_size = 0.1, random_state = 42,
                                     stratify = output_data)

            if self.__labeled_data_balancer is not None and should_balance:
                input_data, output_data = self.__labeled_data_balancer.balance(input_data, output_data)
                if verbose:
                    logging.info(f"Balanced class cardinality: {np.array(to_categorical(output_data)).sum(axis = 0)}")

        return copy.copy((np.array(input_data), np.array(output_data),
                          np.array(input_data_test), np.array(output_data_test)))

    def get_data_for_iteration(self, columns: list[str], start: int = 0, stop: Optional[int] = None,
                               step: int = 1) -> list[float]:
        """
        Data getter for certain iterations.

        Parameters:
            columns (list[str]): List of column names to extract from data.
            start (int): Start iteration index. Defaults to 0.
            stop (int): Stop iteration index. Defaults to environment length minus one.
            step (int): Step between iterations. Defaults to 1.

        Returns:
            (list[float]): Copy of part of data with specified columns
                over specified iterations.
        """

        if stop is None:
            stop = self.get_environment_length() - 1

        return copy.copy(self.__data[self.__mode].loc[start:stop:step, columns].values.ravel().tolist())

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        """
        Performs specified action on environment. It results in generation of the new
        observations. This function causes trades to be handled, reward to be calculated and
        environment to be updated.

        Parameters:
            action (int): Number specifing action. Possible values are 0 for buy action,
                1 for wait action and 2 for sell action.

        Returns:
            (tuple[list[float], float, bool, dict]): Tuple containing next observation
                state, reward, finish indication and additional info dictionary.
        """

        self.current_iteration += 1
        self.state = self.__get_current_state_data()

        close_changes = self.__data[self.__mode].iloc[self.current_iteration - 2 : self.current_iteration]['close'].values
        stock_change_coeff = 1 + (close_changes[1] - close_changes[0]) / close_changes[0]
        closed_orders = self.__broker.update_orders(stock_change_coeff)

        if self.__trading_mode == TradingEnvironment.TradingMode.EXPLICIT_ORDER_CLOSING:
            current_orders = self.__broker.get_current_orders()
            if len(current_orders) > 0:
                was_last_order_placed_as_buy = current_orders[-1].is_buy_order
                if (action == 0 and not was_last_order_placed_as_buy) or \
                    (action == 2 and was_last_order_placed_as_buy):
                    closed_orders += self.__broker.force_close_orders()

        reward = self.__validator.validate_orders(closed_orders)
        self.__trading_data.currently_placed_trades -= len(closed_orders)
        self.__trading_data.current_budget += np.sum([trade.current_value for trade in closed_orders])
        self.__trading_data.currently_invested -= np.sum([trade.initial_value for trade in closed_orders])

        number_of_possible_trades = self.__trading_consts.MAX_AMOUNT_OF_TRADES - self.__trading_data.currently_placed_trades
        money_to_trade = 0
        if number_of_possible_trades > 0:
            money_to_trade = 1.0 / number_of_possible_trades * self.__trading_data.current_budget

        if action == 0:
            is_buy_order = True
            stop_loss = self.__trading_consts.SELL_STOP_LOSS
            take_profit = self.__trading_consts.SELL_TAKE_PROFIT
        elif action == 2:
            is_buy_order = False
            stop_loss = self.__trading_consts.BUY_STOP_LOSS
            take_profit = self.__trading_consts.BUY_TAKE_PROFIT

        if action != 1:
            if number_of_possible_trades > 0:
                self.__trading_data.current_budget -= money_to_trade
                self.__trading_data.currently_invested += money_to_trade
                self.__broker.place_order(money_to_trade, is_buy_order, stop_loss, take_profit)
                self.__trading_data.currently_placed_trades += 1
                self.__trading_data.no_trades_placed_for = 0
                reward += self.__trading_consts.STATIC_REWARD_ADJUSTMENT
            else:
                self.__trading_data.no_trades_placed_for += 1
                reward -= self.__trading_consts.STATIC_REWARD_ADJUSTMENT
        else:
            self.__trading_data.no_trades_placed_for += 1
            if number_of_possible_trades == 0:
                reward += self.__trading_consts.STATIC_REWARD_ADJUSTMENT

        if number_of_possible_trades > 0:
            reward *= (1 - self.__trading_consts.PENALTY_FUNCTION(self.__trading_data.no_trades_placed_for)) \
                      if reward > 0 else 1
            if self.__trading_consts.PENALTY_STOPS < self.__trading_data.no_trades_placed_for:
                reward -= self.__trading_consts.STATIC_REWARD_ADJUSTMENT

        if (self.current_iteration >= self.get_environment_length() - 1 or
            self.__trading_data.current_budget  > 10 * self.__trading_consts.INITIAL_BUDGET or
            (self.__trading_data.current_budget + self.__trading_data.currently_invested) / self.__trading_consts.INITIAL_BUDGET < 0.8):
            done = True
        else:
            done = False

        info = {'coeff': stock_change_coeff,
                'iteration': self.current_iteration,
                'number_of_closed_orders': len(closed_orders),
                'money_to_trade': money_to_trade,
                'action': action,
                'current_budget': self.__trading_data.current_budget,
                'currently_invested': self.__trading_data.currently_invested,
                'no_trades_placed_for': self.__trading_data.no_trades_placed_for,
                'currently_placed_trades': self.__trading_data.currently_placed_trades}

        return self.state, reward, done, info

    def render(self) -> None:
        """
        Renders environment visualization. Will be implemented later.
        """

        #TODO: Visualization to be implemented
        pass

    def reset(self, randkey: Optional[int] = None) -> list[float]:
        """
        Resets environment. Used typically if environemnt is finished,
        i.e. when ther is no more steps to be taken within environemnt
        or finish conditions are fulfilled.

        Parameters:
            randkey (Optional[int]): Value indicating what iteration
                should be trated as starting point after reset.

        Returns:
            (list[float]): Current iteration observation state.
        """

        if randkey is None:
            randkey = random.randint(self.__trading_consts.WINDOW_SIZE, self.get_environment_length() - 1)
        self.__trading_data.current_budget = self.__trading_consts.INITIAL_BUDGET
        self.__trading_data.currently_invested = 0
        self.__trading_data.no_trades_placed_for = 0
        self.__trading_data.currently_placed_trades = 0
        self.__broker.reset()
        self.current_iteration = randkey
        self.state = self.__get_current_state_data()

        return self.state