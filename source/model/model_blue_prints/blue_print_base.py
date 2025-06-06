# model/model_blue_prints/blue_print_base.py

# global imports
import inspect
from abc import ABC, abstractmethod

# local imports
from source.model import ModelAdapterBase

class BluePrintBase(ABC):
    """
    Implements a base class for model blueprints. It provides an interface for
    instantiating models and reporting parameters needed for instantiation.
    """

    @abstractmethod
    def instantiate_model(self, **kwargs) -> ModelAdapterBase:
        """
        Instantiates a model based on the provided keyword arguments.

        Parameters:
            (**kwargs): Keyword arguments containing the parameters needed for model instantiation.

        Returns:
            (ModelAdapterBase): An instance of the model adapter.
        """

        pass

    def report_parameters_needed_for_instantiation(self) -> list[str]:
        """
        Reports the parameters needed for model instantiation.

        Returns:
            (list[str]): A list of parameter names that are required for instantiation.
        """

        instatiate_function_params = dict(inspect.signature(self.instantiate_model).parameters)

        return [name for name, param in instatiate_function_params.items()
                 if param.default is param.empty and param.kind not in [param.VAR_KEYWORD, param.VAR_POSITIONAL]]