# utils/singleton_meta.py

# global imports
from typing import Type, Any

# local imports

class SingletonMeta(type):
    """
    Metaclass for defining singleton classes.
    """

    def __new__(mcs, name: str, bases: tuple, attrs: dict) -> Type:
        """
        Creates a new singleton class type.

        Parameters:
            name (str): The name of the class being created.
            bases (tuple): The base classes of the class being created.
            attrs (dict): The attributes of the class being created.
        """

        cls = super().__new__(mcs, name, bases, attrs)
        cls.__singleton_instance = None
        return cls

    def __call__(cls, *args, **kwargs) -> Any:
        """
        Creates or returns the singleton instance of the class.

        Parameters:
            (*args): Positional arguments for the class constructor
            (**kwargs): Keyword arguments for the class constructor

        Returns:
            (Any): The singleton instance of the class.
        """

        if cls.__singleton_instance is None:
            cls.__singleton_instance = super().__call__(*args, **kwargs)
        return cls.__singleton_instance