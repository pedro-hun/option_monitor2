from datetime import datetime
import numpy as np
import pandas as pd
from typing import Optional


# Function to calculate time to expiration in business days
def calculate_tte_days(current_date: Optional[datetime], expiry: datetime, holidays: pd.DataFrame) -> int:
    """
    Calculate time to expiration in business days for option contracts
    """
    
    # Use provided date or default to today
    if current_date is None:
        today = pd.Timestamp.now().date()
    else:
        today = current_date.date() if hasattr(current_date, 'date') else current_date
    # Convert to datetime if needed
    maturity_date = pd.to_datetime(expiry, format='%d/%m/%Y')

    # Get holidays in the correct format for numpy busday_count
    holidays_for_numpy = pd.to_datetime(holidays["Data"], format='%m/%d/%Y').values.astype('datetime64[D]')

    return int(np.busday_count(today, maturity_date.date(), holidays=holidays_for_numpy))

# Function to calculate time to expiration in years
def calculate_tte_years(tte_days) -> float:
    """
    Calculate time to expiration in years for option contracts
    """

    return tte_days / 252.0

# Function to calculate mid price
def calculate_mid_price(bid_price: Optional[float], ask_price: Optional[float]) -> Optional[float]:
    """
    Calculate the mid price of the option
    """
    if bid_price is not None and ask_price is not None:
        return (bid_price + ask_price) / 2.0
    return None

# Function to calculate bid-ask spread
def calculate_bid_ask_spread(bid_price: Optional[float], ask_price: Optional[float]) -> Optional[float]:
    """
    Calculate the bid-ask spread of the option
    """
    if bid_price is not None and ask_price is not None:
        return ask_price - bid_price
    return None

# Function to calculate bid-ask spread percentage
def calculate_relative_spread(mid_price: Optional[float], spread: Optional[float]) -> Optional[float]:
    """
    Calculate the bid-ask spread percentage of the option
    """
    if mid_price is not None and spread is not None and mid_price != 0:
        return (spread / mid_price) * 100.0
    return None

# Function to calculate forward price
def calculate_forward_price(spot_price: float, risk_free_rate: float, tte_years: float) -> float:
    """
    Calculate the forward price of the underlying asset
    """
    return spot_price * np.power(1.0 + (risk_free_rate), tte_years)

# Function to calculate moneyness
def calculate_moneyness(forward_price: float, strike: float) -> float:
    """
    Calculate the moneyness of the option
    """
    return forward_price / strike

# Function to calculate log moneyness
def calculate_log_moneyness(moneyness) -> float:
    """
    Calculate the log moneyness of the option
    """
    return np.log(moneyness) if moneyness > 0 else -np.inf

# Function to calculate intrinsic value
def calculate_intrinsic_value(forward_price: float, strike: float, option_type: Optional[str]) -> float:
    """
    Calculate the intrinsic value of the option
    """
    if option_type == "call":
        return max(0.0, forward_price - strike)
    elif option_type == "put":
        return max(0.0, strike - forward_price)
    return 0.0

