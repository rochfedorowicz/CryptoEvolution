# model/model_blue_prints/general_sklearn_blue_print.py

# global imports
from sklearn.base import BaseEstimator

# local imports
from source.model import BluePrintBase, ModelAdapterBase, SklearnModelAdapter

class GeneralSklearnBluePrint(BluePrintBase):
    """
    Implements a general blueprint for scikit-learn models. It provides a method to instantiate
    a model based on the provided base estimator class and keyword arguments.
    """

    # local constants
    __VERBOSE_KEY: str = "verbose"

    def __init__(self, base_estimator_class: type, **kwargs) -> None:
        """
        Class constructor. Initializes the blueprint with a base estimator class and optional keyword arguments.

        Parameters:
            base_estimator_class (type): The base estimator class to use for model instantiation.
            (**kwargs): Optional keyword arguments for the model.
        """

        if not issubclass(base_estimator_class, BaseEstimator):
            raise TypeError(
                f"Parameter base_estimator_class must be a subclass of BaseEstimator,"
                f" got {base_estimator_class.__name__}"
            )

        self.__base_estimator_class = base_estimator_class
        self.__kwargs = kwargs

        if self.__VERBOSE_KEY not in self.__kwargs:
            self.__kwargs[self.__VERBOSE_KEY] = True

    def instantiate_model(self, **kwargs) -> ModelAdapterBase:
        """
        Instantiates a model based on the provided keyword arguments.

        Parameters:
            (**kwargs): Optional keyword arguments for the model.

        Returns:
            (ModelAdapterBase): The model adapter for the instantiated model.
        """

        if len(kwargs.keys()) == 0:
            kwargs = self.__kwargs

        return SklearnModelAdapter(self.__base_estimator_class(**kwargs))