# environment/labeled_data_balancer.py

# global imports
from imblearn.base import BaseSampler

# local imports

class LabeledDataBalancer:
    """
    Implements a labeled data balancer that uses a list of samplers to balance the input and output data.
    """

    def __init__(self, balancers: list[BaseSampler]) -> None:
        """
        Class constructor. Initializes the balancer with a list of samplers.

        Parameters:
            balancers (list[BaseSampler]): A list of samplers to be used for balancing the data.
        """

        self.__balancers = balancers

    def balance(self, input_data: list[list[float]], output_data: list[int]) -> tuple[list[list[float]], list[int]]:
        """
        Balances the input and output data using the configured samplers.

        Parameters:
            input_data (list[list[float]]): The input data to be balanced.
            output_data (list[int]): The output data to be balanced.

        Raises:
            ValueError: If the input data and output data do not have the same length.

        Returns:
            (tuple[list[list[float]], list[int]]): A tuple containing the balanced input data and output data.
        """

        if len(input_data) != len(output_data):
            raise ValueError("Input data and output data must have the same length.")

        for balancer in self.__balancers:
            input_data, output_data = balancer.fit_resample(input_data, output_data)

        return input_data, output_data