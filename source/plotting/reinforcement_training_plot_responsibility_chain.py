# plotting/reinforcement_training_plot_responsibility_chain.py

# global imports
import matplotlib.pyplot as plt
import pandas as pd

# local imports
from source.agent import ReinforcementLearningStrategyHandler
from source.plotting import PlotResponsibilityChainBase

class ReinforcementTrainingPlotResponsibilityChain(PlotResponsibilityChainBase):
    """
    Implements a plotting responsibility chain for reinforcement learning training results.
    It implements the _can_plot and _plot methods to visualize training history.
    """

    def __init__(self, window_size: int = 5) -> None:
        """
        Class constructor. Initializes the plotting responsibility chain with a window size.

        Parameters:
            window_size (int): The size of the moving average window for smoothing the rewards.
        """

        self.__window_size = window_size

    def _can_plot(self, key: str) -> bool:
        """
        Checks if the plot can be generated for the given key.

        Parameters:
            key (str): The key to check.

        Returns:
            (bool): True if the plot can be generated, False otherwise.
        """

        return key == ReinforcementLearningStrategyHandler.PLOTTING_KEY

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates the plot for the reinforcement learning training results.

        Parameters:
            plot_data (dict): The data to be plotted.

        Returns:
            (plt.Axes): The axes object containing the plot.
        """

        steps = [0] + plot_data['nb_steps']
        reward = [0] + plot_data['episode_reward']
        adjusted_window_size = min(self.__window_size, len(reward))
        avg_reward = pd.Series(reward).rolling(
            window = adjusted_window_size,
            min_periods = 1
        ).mean().values

        plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        plt.plot(steps, reward, label = 'Episode reward', color = 'blue')
        plt.plot(steps, avg_reward,
                 label = f'{adjusted_window_size}-step moving average of episode reward', color = 'red')
        plt.title('Training history')
        plt.xlabel('Trading points')
        plt.ylabel('Reward')
        plt.legend()

        return plt.gca()
