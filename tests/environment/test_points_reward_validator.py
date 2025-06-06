# tests/environment/test_points_reward_validator.py

# global imports
import logging
from ddt import data, ddt, unpack
from unittest import TestCase

# local imports
from source.environment import Order, PointsRewardValidator

@ddt
class PointsRewardValidatorTestCase(TestCase):
    """
    Test case PointsRewardValidator class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.validator = PointsRewardValidator()

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    @data(
        ([Order(100, True, 0.8, 1.2)], [125], 1),
        ([Order(100, False, 0.8, 1.2)], [75], -1),
        ([Order(100, True, 0.9, 1.3), Order(100, False, 0.9, 1.3)], [135, 85], 0)
    )
    @unpack
    def test_points_reward_validator_validate_orders(self, orders: list[Order], current_values: list[float],
                                                     expected_reward: int) -> None:
        """
        Tests PointsRewardValidator's validate orders functionality.

        Verifies that policy used for points calculation works in
        accordance with the assumptions.

        Asserts:
            Reward equals expected reward.
        """

        logging.info("Starting validate orders test case.")
        for order, new_current_value in zip(orders, current_values):
            order.current_value = new_current_value

        reward = self.validator.validate_orders(orders)

        logging.info("Checking expected reward.")
        assert reward == expected_reward