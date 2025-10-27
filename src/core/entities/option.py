from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

@dataclass
class Option:
    """Core option entity - represents a single option contract"""
    strike: float
    expiry: datetime
    option_type: OptionType
    underlying: str
    spot_price: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    last_price: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None


# Function to get option type
    def get_option_type(self) -> OptionType:
        """
        Get the option type (call or put)
        """
        calls = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        puts = ["M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]
        if self.underlying[4] in calls:
            self.option_type = OptionType.CALL
        elif self.underlying[4] in puts:
            self.option_type = OptionType.PUT
        else :
            raise ValueError("Invalid option type in underlying symbol")
        return self.option_type