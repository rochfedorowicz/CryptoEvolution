# plotting/plot_responsibility_chain_base.py

# global imports
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from typing import Optional

# local imports

class PlotResponsibilityChainBase(ABC):
    """
    Base class for implementing the responsibility chain pattern for plotting.

    This class provides a framework for creating a chain of plotting handlers where
    each handler can decide whether it can handle a specific plot request. If a handler
    cannot process the request, it passes the request to the next handler in the chain.
    All plotting chain implementations should inherit from this class and implement
    the _can_plot and _plot methods.
    """

    # local constants
    __KEY_STR: str = 'key'
    __PLOT_DATA_STR: str = 'plot_data'
    __next_chain_link: 'PlotResponsibilityChainBase' = None

    # derived constants
    _EXPECTED_FIGURE_SIZE = (int(letter[0] // inch - 1), int(letter[1] // inch - 2))

    def plot(self, data) -> Optional[plt.Axes]:
        """
        Attempts to plot data by finding an appropriate handler in the chain.

        Parameters:
            data (dict): Dictionary containing 'key' identifying the plot type
                        and 'plot_data' containing the actual data to be plotted.

        Returns:
            Optional[plt.Axes]: The matplotlib Axes object if plotting was successful,
                               None if no handler in the chain could process the request.
        """

        key = data[self.__KEY_STR]
        plot_data = data[self.__PLOT_DATA_STR]

        if (self._can_plot(key)):
            return self._plot(plot_data)
        else:
            if self.__next_chain_link is not None:
                return self.__next_chain_link.plot(data)
            else:
                return None

    def add_next_chain_link(self, next_chain_link: 'PlotResponsibilityChainBase'):
        """
        Adds the next handler to the responsibility chain.

        Parameters:
            next_chain_link (PlotResponsibilityChainBase): The next handler to process
                                                          requests if this handler cannot.
        """

        current_last_chain_link = self
        while current_last_chain_link.__next_chain_link is not None:
            current_last_chain_link = current_last_chain_link.__next_chain_link

        current_last_chain_link.__next_chain_link = next_chain_link

    @abstractmethod
    def _can_plot(self, key: str) -> bool:
        """
        Determines if this handler can plot the given plot type.

        This is an abstract method that should be implemented by all subclasses.

        Parameters:
            key (str): String identifier of the plot type.

        Returns:
            bool: True if this handler can process the plot request, False otherwise.
        """
        pass

    @abstractmethod
    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates the actual plot from the provided data.

        This is an abstract method that should be implemented by all subclasses.
        The implementation should create and return a matplotlib plot based on
        the provided data.

        Parameters:
            plot_data (dict): Dictionary containing the data needed for plotting.

        Returns:
            plt.Axes: Matplotlib Axes object containing the generated plot.
        """

        pass