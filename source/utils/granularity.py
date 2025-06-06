# utils/granularity.py

# global imports
from enum import Enum

# local imports

class Granularity(Enum):
    """
    Enum representing possible values of time granularity that coinbase API can handle.
    Values assigned to particular options represent number of seconds that certain option denotes.
    """

    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    THIRTY_MINUTES = 1800
    ONE_HOUR = 3600
    SIX_HOURS = 21600
    ONE_DAY = 86400