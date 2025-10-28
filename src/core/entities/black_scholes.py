import numpy as np
from scipy.stats import norm

def w(iv: float, tte_years: float) -> float:
    """
    Auxiliary function to compute w from BSM model.
    @param iv: Implied volatility
    @param tte_years: Time to expiration in years
    @return: Values of w
    """
    return iv * iv * tte_years

def d1(w, k):
    """
    Auxiliary function to compute d1 from BSM model.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Values of d1 evaluated at k points
    """
    v = np.sqrt(w)
    return -k/v + 0.5*v


def d2(w, k):
    """
    Auxiliary function to compute d2 from BSM model.
    @param k: Moneyness points to evaluate
    @return: Values of d2 evaluated at k points
    """
    v = np.sqrt(w)
    return -k/v - 0.5*v

def discount_factor(r: float, tte_years: float) -> float:
    """
    Calculate discount factor
    @param r: Risk-free interest rate
    @param tte_years: Time to expiration in years
    """
    return (1 + r) ** (-tte_years)

def black_price_forward(forward_price: float, strike: float, tte_years: float, iv: float, r: float, option_type: str, k: float) -> float:
    """
    Calculate Black-Scholes price from forward price and implied volatility
    """
    DF = discount_factor(r, tte_years)
    w_val = w(iv, tte_years)
    if option_type == "call":
        price = DF * ((forward_price * norm.cdf(d1(w_val, k)) - strike * norm.cdf(d2(w_val, k))))
        return float(price)
    elif option_type == "put":
        price = DF * (-(forward_price * norm.cdf(-d1(w_val, k)) - strike * norm.cdf(-d2(w_val, k))))
        return float(price)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")
    
