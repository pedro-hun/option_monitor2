from src.core.entities.black_scholes import d1, d2, discount_factor
from scipy.stats import norm
import numpy as np

def vega(spot_price: float, tte_years: float, d1: float, borrow: float = 0) -> float:
    """
    Calculate Black-Scholes vega from spot price and d1
    """

    return spot_price * np.exp(-borrow * tte_years) * np.sqrt(tte_years) * norm.pdf(d1)

def theta(spot_price: float, strike: float, tte_years: float, sigma: float, 
          d1: float, d2: float, r: float, option_type: str, borrow: float) -> float:
    """
    Calculate Black-Scholes theta (time decay)
    """
    
    p = np.log(1 + r)
    
    # Common term for both calls and puts
    term1 = (-spot_price * np.exp(-borrow * tte_years) * norm.pdf(d1) * sigma) / (2 * np.sqrt(tte_years))
    
    # Second term differs for calls and puts
    if option_type == 'call':
        term2 = borrow * spot_price * np.exp(-borrow * tte_years) * norm.cdf(d1) - p * strike * discount_factor(r, tte_years) * norm.cdf(d2)
    elif option_type == 'put':
        term2 = - borrow * spot_price * np.exp(-borrow * tte_years) * norm.cdf(d1) + p * strike * discount_factor(r, tte_years) * norm.cdf(-d2)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")        
    return term1 + term2

def rho(strike: float, tte_years: float, d2: float, r: float, option_type: str) -> np.ndarray:
    """
    Calculate Black-Scholes rho (interest rate sensitivity)
    """

    if option_type == 'call':
        dC_dp = tte_years * strike * discount_factor(r, tte_years) * norm.cdf(d2)

        return dC_dp / (1 + r)
    elif option_type == 'put':
        dP_dp = -tte_years * strike * discount_factor(r, tte_years) * norm.cdf(-d2)
        return dP_dp / (1 + r)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")
    
def delta(tte_years: float, d1: float, option_type: str, borrow: float) -> float:
    """
    Calculate Black-Scholes delta from d1
    """

    if option_type == 'call':
        delta_val = np.exp(-borrow * tte_years) * norm.cdf(d1)
    elif option_type == 'put':
        delta_val = np.exp(-borrow * tte_years) * (norm.cdf(d1) - 1)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")
    
    return delta_val

def gamma(spot_price: float, tte_years: float, sigma: float, d1: float, borrow: float) -> float:
    """
    Calculate Black-Scholes gamma from spot price and d1
    """

    return (np.exp(-borrow * tte_years) * norm.pdf(d1)) / (spot_price * sigma * np.sqrt(tte_years))