# environment/points_reward_validator.py

# global imports

# local imports
from source.environment import Order, RewardValidatorBase

class PointsRewardValidator(RewardValidatorBase):
    """
    Awards reward for successful or failure order basing on predefined constants.
    """

    def __init__(self, rewarded_points: tuple[int, int] = (1, -1)) -> None:
        """
        Class constructor.

        Parameters:
            rewarded_points (tuple[int, int]): Number of points to be awarded for
                successful and failure trade respectively.
        """

        self.__rewarded_points: tuple[int, int] = rewarded_points

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
            if order.current_value > order.initial_value:
                reward = reward + self.__rewarded_points[0]
            else:
                reward = reward + self.__rewarded_points[1]
        return reward