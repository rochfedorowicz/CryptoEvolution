# environment/reward_validator_base.py

# global imports
from abc import ABC, abstractmethod

# local imports
from source.environment import Order

class RewardValidatorBase(ABC):
    """
    Awards reward for successful or failure order basing on approach defined in
    derivative class.
    """

    @abstractmethod
    def __init__(self, *args) -> None:
        """
        Class constructor. Parameters are specified in derivative classes.
        """

        pass

    @abstractmethod
    def validate_orders(self, orders: list[Order]) -> float:
        """
        Calculates number of points to be rewarded for list of closed trades.

        Parameters:
            orders (list[Order]): Orders to be validated.

        Returns:
            (float): Calcualted reward.
        """

        pass