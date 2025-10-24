from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
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
    current_date: Optional[datetime] = None
    risk_free_rate: float = 0.15

# Function to calculate time to expiration in business days
    def calculate_tte_business_days(self) -> int:
        """
        Calculate time to expiration in business days for option contracts
        """
        
        # Use provided date or default to today
        if self.current_date is None:
            today = pd.Timestamp.now().date()
        else:
            today = self.current_date.date() if hasattr(self.current_date, 'date') else self.current_date
        # Convert to datetime if needed
        maturity_date = pd.to_datetime(self.expiry, format='%d/%m/%Y')

        # Load holidays from CSV
        holidays_df = pd.read_csv('feriados_nacionais.csv')

        # Get holidays in the correct format for numpy busday_count
        holidays_for_numpy = pd.to_datetime(holidays_df["Data"], format='%m/%d/%Y').values.astype('datetime64[D]')

        return int(np.busday_count(today, maturity_date.date(), holidays=holidays_for_numpy))
    
# Function to calculate time to expiration in years
    def calculate_tte_years(self) -> float:
        """
        Calculate time to expiration in years for option contracts
        """

        return self.calculate_tte_business_days() / 252.0
    
# Function to calculate mid price
    def calculate_mid_price(self) -> Optional[float]:
        """
        Calculate the mid price of the option
        """
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2.0
        return None
    
# Function to calculate moneyness
    def calculate_moneyness(self) -> float:
        """
        Calculate the moneyness of the option
        """
        return self.spot_price / self.strike

# Function to calculate log moneyness
    def calculate_log_moneyness(self) -> float:
        """
        Calculate the log moneyness of the option
        """
        moneyness = self.calculate_moneyness()
        return np.log(moneyness) if moneyness > 0 else -np.inf
    
# Function to calculate forward price
    def calculate_forward_price(self) -> float:
        """
        Calculate the forward price of the underlying asset
        """
        tte_years = self.calculate_tte_years()
        return self.spot_price * np.power(1.0 + (self.risk_free_rate), tte_years)
    
# Function to calculate intrinsic value
    def calculate_intrinsic_value(self) -> float:
        """
        Calculate the intrinsic value of the option
        """
        if self.option_type == OptionType.CALL:
            return max(0.0, self.calculate_forward_price() - self.strike)
        elif self.option_type == OptionType.PUT:
            return max(0.0, self.strike - self.calculate_forward_price())
        return 0.0
    
