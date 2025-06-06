# tests/environment/test_broker.py

# global imports
import logging
from ddt import data, ddt, unpack
from unittest import TestCase

# local imports
from source.environment import Broker, Order

@ddt
class BrokerTestCase(TestCase):
    """
    Test case Broker class. Stores all the test cases and allows for
    convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.broker = Broker()

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
            for attribute_name in self.broker.__dict__:
                if name in attribute_name:
                    setattr(self.broker, attribute_name, value)

    @data(
        [(50.5, False, 0.8, 1.2)],
        [(102.5, True, 0.9, 1.3)],
        [(50.5, False, 0.8, 1.2), (102.5, True, 0.9, 1.3)]
    )
    def test_broker_place_order(self, list_of_order_arguments: list[tuple[float, bool, float, float]]) -> None:
        """
        Tests Broker's place order functionality.

        Verifies that all the placed orders were created exactly with
        exactly the same parameters as it was intended, and were not changed
        later.

        Asserts:
            Length of broker current orders matches number of trades to be
            placed in a test case. Also, for each order all the members are
            checked with original parameters passed to place order function.
        """

        logging.info("Starting place order test case.")
        for order_argument in list_of_order_arguments:
            self.broker.place_order(*order_argument)

        logging.info("Checking created orders.")
        current_orders = self.broker.get_current_orders()
        assert len(current_orders) == len(list_of_order_arguments)
        for placed_order, order_argument in zip(current_orders, list_of_order_arguments):
            amount, is_buy_order, stop_loss, take_profit = order_argument
            assert placed_order.initial_value == amount
            assert placed_order.current_value == amount
            assert placed_order.is_buy_order == is_buy_order
            assert placed_order.stop_loss == stop_loss
            assert placed_order.take_profit == take_profit

    @data(
        ([Order(100, True, 0.8, 1.2)], 1.3, 1, 1),
        ([Order(100, False, 0.8, 1.2)], 1.3, 1, 1),
        ([Order(100, True, 0.9, 1.3), Order(100, False, 0.9, 1.3)], 0.8, 1, 1),
        ([Order(100, True, 0.9, 1.3), Order(100, False, 0.9, 1.3)], 1.2, 2, 2),
    )
    @unpack
    def test_broker_update_orders(self, orders: list[Order], coeff: float, leverage: float,
                                  expected_closed_trades: int) -> None:
        """
        Tests Broker's update orders functionality.

        Verifies that all orders that should be closed were closed correctly
        and remaining orders were increased or decreased in value correctly.

        Asserts:
            Length of broker closed trades equals expected length of closed
            trades and current value of remaining orders was calculated properly.
        """

        logging.info("Starting update orders test case.")
        self.__update_sut(leverage = leverage,
                          current_orders = orders)
        closed_trades = self.broker.update_orders(coeff)

        logging.info("Checking updated orders.")
        assert len(closed_trades) == expected_closed_trades
        for order in self.broker.get_current_orders():
            order_change = order.current_value - order.initial_value
            expected_change = order.initial_value * (coeff - 1) * leverage \
                                                * (1 if order.is_buy_order else -1)
            assert round(order_change, 5) == round(expected_change, 5)

    def test_broker_reset(self) -> None:
        """
        Tests Broker's reset functionality.

        Verifies that all orders were correctly cleared from broker after reset.

        Asserts:
            Length of broker current orders before reset is not equal to 0 and
            length of broker current orders after reset is equal to 0.
        """

        logging.info("Starting reset test case.")
        orders = [Order(100, True, 0.9, 1.3), Order(100, False, 0.9, 1.3)]
        self.__update_sut(current_orders = orders)

        initial_orders_length = len(self.broker.get_current_orders())
        self.broker.reset()
        final_orders_length = len(self.broker.get_current_orders())

        logging.info("Checking orders list after reset.")
        assert initial_orders_length != 0
        assert final_orders_length == 0