# environment/order.py

# global imports

# local imports

class Order():
    """
    Class storing information regarding particular order.
    """

    def __init__(self, amount: float, is_buy_order: bool, stop_loss: float, take_profit: float) -> None:
        """
        Class constructor.

        Parameters:
            amount (float): Amount of money assigned to order.
            is_buy_order (bool): Indicates whether order should be treated as buy (long)
                position - when true - or as sell (short) postion - when false
            stop_loss (float): Coefficient used to close order when stock behaves contrary
                to expectations.
            take_profit (float): Coefficient used to close order when stock behaves accordingly
                to expectations.
        """

        self.initial_value: float = amount
        self.current_value: float = amount
        self.is_buy_order: bool = is_buy_order
        self.stop_loss: float = stop_loss
        self.take_profit: float = take_profit