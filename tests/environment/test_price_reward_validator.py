# tests/environment/test_price_reward_validator.py

# global imports
import logging
from ddt import data, ddt, unpack
from unittest import TestCase

# local imports
from source.environment import Order, PriceRewardValidator

@ddt
class PriceRewardValidatorTestCase(TestCase):
    """
    Test case PriceRewardValidator class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.validator = PriceRewardValidator(1.0, False)

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    def __update_sut(self, **kwargs) -> None:
        """
        Allows to update already created sut. It speeds up test
        cases' scenarios by enabling injecting certain values
        also into private sut members.
        """

        for name, value in kwargs.items():
            for attribute_name in self.validator.__dict__:
                if name in attribute_name:
                    setattr(self.validator, attribute_name, value)

    @data(
        ([Order(100, True, 0.8, 1.2)], [125], 25, False),
        ([Order(100, False, 0.8, 1.2)], [75], -25, True),
        ([Order(90, True, 0.9, 1.3), Order(100, False, 0.9, 1.3)], [135, 85], 30, False),
        ([Order(90, True, 0.9, 1.3), Order(100, False, 0.9, 1.3)], [135, 85], 35, True)
    )
    @unpack
    def test_price_reward_validator_validate_orders(self, orders: list[Order], current_values: list[float],
                                                    expected_reward: int, normalizable: bool) -> None:
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

        self.__update_sut(normalizable = normalizable)
        reward = self.validator.validate_orders(orders)

        logging.info("Checking expected reward.")
        assert reward == expected_reward