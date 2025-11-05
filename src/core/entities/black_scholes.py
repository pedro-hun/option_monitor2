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

def d1(w: float, k: float) -> float:
    """
    Auxiliary function to compute d1 from BSM model.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Values of d1 evaluated at k points
    """

    v = np.sqrt(w)
    return -k/v + 0.5*v


def d2(w: float, k: float) -> float:
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

def black_scholes(spot_price: float, tte_years: float, iv: float, option_type: str, k: float) -> float:
    """
    Calculate Black-Scholes price from forward price and implied volatility
    """
    w_val = w(iv, tte_years)
    if option_type == "call":
        price = spot_price *  (norm.cdf(d1(w_val, k)) - np.exp(k) * norm.cdf(d2(w_val, k)))
    elif option_type == "put":
        price = spot_price * (np.exp(k) * norm.cdf(-d2(w_val, k)) - norm.cdf(-d1(w_val, k)))
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")
    return price

if __name__ == "__main__":
    import pandas as pd

    def soma (a: float, b: float) -> float:
        return a + b

    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([4.0, 5.0, 6.0])
    df = pd.DataFrame({'bid': a, 'ask': b})
    soma_series = df.apply(lambda row: soma(row['bid'], row['ask']), axis=1)
    print(soma_series)