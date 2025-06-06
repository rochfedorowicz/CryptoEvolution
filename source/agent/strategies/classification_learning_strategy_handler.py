# agent/strategies/classification_learning_strategy_handler.py

# global imports
import logging
from tensorflow.keras.callbacks import Callback
from typing import Any

# local imports
from source.agent import AgentBase, ClassificationLearningAgent, LearningStrategyHandlerBase
from source.environment import TradingEnvironment
from source.model import BluePrintBase

class ClassificationLearningStrategyHandler(LearningStrategyHandlerBase):
    """
    Implements a learning strategy handler for classification tasks. It provides
    functionalities for creating agents and fitting models.
    """

    # global class constants
    PLOTTING_KEYS: str = ['price_movement_trend_class_summary', 'classification_learning']

    def create_agent(self, model_blue_print: BluePrintBase,
                     trading_environment: TradingEnvironment) -> AgentBase:
        """
        Creates a classification learning agent. It dynamically determines the parameters
        needed for instantiation based on the model blueprint.

        Parameters:
            model_blue_print (BluePrintBase): The model blueprint to be used for agent creation.
            trading_environment (TradingEnvironment): The trading environment in which the agent will operate.

        Returns:
            (Agent): An instance of the classification learning agent created using the model blueprint and trading environment.
        """

        parameters_needed_for_instantiation = model_blue_print.report_parameters_needed_for_instantiation()
        kwargs = {}
        for parameter in parameters_needed_for_instantiation:
            kwargs[parameter] = self._provide_required_parameter(parameter, trading_environment)

        return ClassificationLearningAgent(model_blue_print.instantiate_model(**kwargs))

    def fit(self, agent: ClassificationLearningAgent, trading_environment: TradingEnvironment,
            nr_of_steps: int, nr_of_episodes: int, callbacks: list[Callback]) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Fits the classification learning agent to the trading environment.

        Parameters:
            agent (ClassificationLearningAgent): The classification learning agent to fit.
            environment (TradingEnvironment): The trading environment to use.
            nr_of_steps (int): The number of training steps to perform.
            nr_of_episodes (int): The number of training episodes to perform.
            callbacks (list[Callback]): List of callbacks to use during training.

        Raises:
            TypeError: If the agent is not an instance of ClassificationLearningAgent.

        Returns:
            (list[str], list[dict[str, Any]]): A tuple containing the keys and data collected during training.
        """

        if not isinstance(agent, ClassificationLearningAgent):
            raise TypeError("Agent must be an instance of ClassificationLearningAgent.")

        keys, data = super().fit(agent, trading_environment, nr_of_steps, nr_of_episodes, callbacks)
        keys.append(self.PLOTTING_KEYS[0])
        keys.append(self.PLOTTING_KEYS[1] + "_" + agent.get_model_adapter_tag())

        input_data, output_data, input_data_test, output_data_test = trading_environment.get_labeled_data()
        steps_per_epoch = nr_of_steps // nr_of_episodes
        batch_size = len(input_data) // steps_per_epoch
        validation_tuple = (input_data_test, output_data_test)
        if batch_size <= 0:
            logging.warning("Batch size is zero or negative, using value of 1 instead.")
            batch_size = 1

        data.append(self.__prepare_price_movement_trend_class_summary_plot_data(trading_environment))
        data.append(agent.classification_fit(input_data, output_data, validation_data = validation_tuple,
                                             batch_size = batch_size, epochs = nr_of_episodes, callbacks = callbacks))

        return keys, data

    def _provide_input_shape(self, trading_environment: TradingEnvironment) -> tuple[int, int]:
        """
        Provides the input shape for the model based on the trading environment.

        Parameters:
            trading_environment (TradingEnvironment): The trading environment to use.

        Returns:
            (tuple[int, int]): The input shape for the model.
        """

        windows_size = trading_environment.get_trading_consts().WINDOW_SIZE
        spatial_data_shape = trading_environment.get_environment_spatial_data_dimension()
        return (spatial_data_shape[1] * windows_size, )

    def _provide_output_length(self, trading_environment: TradingEnvironment) -> int:
        """
        Provides the output length for the model based on the trading environment.

        Parameters:
            trading_environment (TradingEnvironment): The trading environment to use.

        Returns:
            (int): The output length for the model.
        """

        return len(trading_environment.get_trading_consts().OUTPUT_CLASSES)

    def _provide_spatial_data_shape(self, trading_environment: TradingEnvironment) -> tuple[int, int]:
        """
        Provides the spatial data shape for the model based on the trading environment.

        Parameters:
            trading_environment (TradingEnvironment): The trading environment to use.

        Returns:
            (tuple[int, int]): The spatial data shape for the model.
        """

        return trading_environment.get_environment_spatial_data_dimension()

    def __prepare_price_movement_trend_class_summary_plot_data(self, trading_environment: TradingEnvironment) -> \
        dict[str, Any]:
        """
        Prepares the data for the price movement trend classification summary plot.

        Parameters:
            environment (TradingEnvironment): The trading environment to use.

        Returns:
            (dict[str, Any]): The prepared data for the plot.
        """

        data = {}
        trading_environment.set_mode(TradingEnvironment.TEST_MODE)
        data['test_part_price_movement'] = trading_environment.get_data_for_iteration(['close'])
        _, test_part_labels, _, _ = trading_environment.get_labeled_data(should_split = False,
                                                                 should_balance = False,
                                                                 verbose = False)
        data['test_part_labels'] = test_part_labels.astype('int')

        trading_environment.set_mode(TradingEnvironment.TRAIN_MODE)
        data['train_part_price_movement'] = trading_environment.get_data_for_iteration(['close'])
        _, train_part_labels, _, _ = trading_environment.get_labeled_data(should_split = False,
                                                                       should_balance = False,
                                                                       verbose = False)
        data['train_part_labels'] = train_part_labels.astype('int')

        return data