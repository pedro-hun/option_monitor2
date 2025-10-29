def option_data_positive(bid, ask, tte_years, iv):
    """
    Check if option data values are positive and not NaN
    """
    if bid is None or bid <= 0 or ask is None or ask <= 0 or tte_years is None or tte_years <= 0 or iv is None or iv <= 0:
        return False
    return True

def intrinsic_lower_than_price(intrinsic_value, mid_price):
    """
    Check if intrinsic value is lower than mid price
    """
    if intrinsic_value is not None and mid_price is not None and intrinsic_value < mid_price:
        return True
    return False

def option_lower_than_underlying(mid, spot_price, option_type):
    """
    Check if option is bigger than underlying price based on option type
    """
    if option_type == "call" and mid <= spot_price:
        return True
    elif option_type == "put" and mid >= spot_price:
        return True
    return False

def valid_price(market_price, price_at_low_vol, price_at_high_vol):
    """
    Check if implied volatility is within valid range
    """
    if market_price is not None and price_at_low_vol is not None and price_at_high_vol is not None:
        if price_at_low_vol <= market_price <= price_at_high_vol:
            return True
    return False

def valid_obj_func(obj_low, obj_high):
    """
    Check if objective function values have opposite signs
    """
    if obj_low or obj_high > 1e9:
        return False
    return True

def valid_iv(iv, low_vol, high_vol):
    """
    Check if implied volatility is within valid range
    """
    if iv is not None and low_vol < iv < high_vol:
        return True
    return False