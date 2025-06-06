# environment/broker.py

# global imports
import copy

# local imports
from source.environment import Order

class Broker():
    """
    Responsible for managing and placing orders. It implements
    real-environment behavior of orders price actions.
    """

    def __init__(self, leverage: int = 1) -> None:
        """
        Class constructor.

        Parameters:
            leverage (int): Coefficient of multiplication used to simulate
                leverage trading.
        """

        self.__leverage: int = leverage
        self.__current_orders: list[Order] = []
        self.__recently_closed_orders: list[Order] = []

    def get_leverage(self) -> int:
        """
        Leverage getter.

        Returns:
            (int): Copy of leverage coefficient.
        """

        return copy.copy(self.__leverage)

    def get_current_orders(self) -> list[Order]:
        """
        Current orders getter.

        Returns:
            (list[Order]): Copy of the list of currently ongoing trades.
        """

        return copy.copy(self.__current_orders)

    def place_order(self, amount: float, is_buy_order: bool, stop_loss: float, take_profit: float) -> None:
        """
        Creates trade with given parameters and attach it to current broker's orders.

        Parameters:
            amount (float): Amount of money assigned to order.
            is_buy_order (bool): Indicates whether order should be treated as buy (long)
                position - when true - or as sell (short) postion - when false
            stop_loss (float): Coefficient used to close order when stock behaves contrary
                to expectations.
            take_profit (float): Coefficient used to close order when stock behaves accordingly
                to expectations.
        """

        self.__current_orders.append(Order(amount, is_buy_order, stop_loss, take_profit))

    def update_orders(self, coefficient: float) -> list[Order]:
        """
        Updates and closes orders. The current value of the order is multiplied by
        coefficient and if stop loss or take profit boundaries are crossed, then
        the order is closed.

        Parameters:
            coefficient (float): Multiplyer that current order value is multiplied by.

        Returns:
            (list[Order]): List of closed trades.
        """

        self.__recently_closed_orders.clear()
        buy_trade_coefficient = ((coefficient - 1) * self.__leverage) + 1
        sell_trade_coefficient = 2 - buy_trade_coefficient

        for order in self.__current_orders:
            if order.is_buy_order:
                order.current_value = order.current_value * buy_trade_coefficient
            else:
                order.current_value = order.current_value * sell_trade_coefficient

            order_ratio = order.current_value / order.initial_value
            if order_ratio >= order.take_profit or order_ratio <= order.stop_loss:
                self.__recently_closed_orders.append(order)

        for order in self.__recently_closed_orders:
            self.__current_orders.remove(order)

        return self.__recently_closed_orders

    def reset(self) -> None:
        """
        Resets broker by clearing the currently ongoing and recently closed
        lists of orders.
        """

        self.__current_orders.clear()
        self.__recently_closed_orders.clear()