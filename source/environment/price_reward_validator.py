# environment/price_reward_validator.py

# global imports

# local imports
from source.environment import Order, RewardValidatorBase

class PriceRewardValidator(RewardValidatorBase):
    """
    Awards reward for successful or failure order basing on gained or lost value.
    """

    def __init__(self, coefficient: float = 1.0, normalizable: bool = False) -> None:
        """
        Class constructor.

        Parameters:
            coefficient (float): Coefficient to be multiplied with gained or lost
                order value.
            normalizable (bool): Indicates if calculated reward should be divided
            by initial order value. Normalizable approach ensures that bigger orders
            contribute the same way to the reward as smaller orders.
        """

        self.__coefficient: float = coefficient
        self.__normalizable: bool = normalizable

    def validate_orders(self, orders: list[Order]) -> float:
        """
        Calculates number of points to be rewarded for list of closed trades.

        Parameters:
            orders (list[Order]): Orders to be validated.

        Returns:
            (float): Calcualted reward.
        """

        reward = 0
        for order in orders:
            summand = (order.current_value - order.initial_value) * self.__coefficient
            if (self.__normalizable):
                summand = summand / order.initial_value * 100
            reward = reward + summand
        return reward