# utils/granularity.py

# global imports
from enum import Enum

# local imports

class Granularity(Enum):
    """
    Enum representing possible values of time granularity that coinbase API can handle.
    Values assigned to particular options represent number of seconds that certain option denotes.
    """

    # global class constants
    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    THIRTY_MINUTES = 1800
    ONE_HOUR = 3600
    SIX_HOURS = 21600
    ONE_DAY = 86400

    def __str__(self) -> str:
        """
        Returns string representation of the granularity.

        Returns:
            (str): String representation of the granularity.
        """

        map = {
            Granularity.ONE_MINUTE: "1m",
            Granularity.FIVE_MINUTES: "5m",
            Granularity.FIFTEEN_MINUTES: "15m",
            Granularity.THIRTY_MINUTES: "30m",
            Granularity.ONE_HOUR: "1h",
            Granularity.SIX_HOURS: "6h",
            Granularity.ONE_DAY: "1d"
        }

        return map[self]

    @classmethod
    def from_string(cls, granularity_str: str) -> 'Granularity':
        """
        Converts string representation of granularity to Granularity enum.

        Parameters:
            granularity_str (str): String representation of granularity.

        Returns:
            Granularity: Corresponding Granularity enum.
        """

        for granularity in cls:
            if granularity_str == str(granularity):
                return granularity
        raise ValueError(f"Invalid granularity string: {granularity_str}")