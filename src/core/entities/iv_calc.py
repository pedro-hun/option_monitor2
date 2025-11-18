from dataclasses import dataclass
import warnings
from typing import Any, Optional
import numpy as np
from scipy.optimize import brentq
from src.core.entities.black_scholes import black_scholes, black_scholes_og
from src.filters.option_filters import valid_price, valid_obj_func, valid_iv

tol = 1e-6
LOW_VOL = 1e-4
HIGH_VOL = 4.0
BORROW = 0.0

class IVCalc:
    def __init__(self, market_price: float, forward_price: float, spot_price: float, strike: float, tte_years: float, r: float, option_type: str, k: float, sigma: float):
        self.sigma = sigma
        self.market_price = market_price
        self.forward_price = forward_price
        self.spot_price = spot_price
        self.strike = strike
        self.tte_years = tte_years
        self.r = r
        self.option_type = option_type
        self.k = k
        self.low_vol = LOW_VOL
        self.high_vol = HIGH_VOL

    def objective_func(self, vol) -> float:
        # Return large value for invalid sigma to guide solver
        if vol <= 0:
            print("Invalid sigma value")
            return 1e10
        model_price = black_scholes(spot_price=self.spot_price, tte_years=self.tte_years, iv=vol, option_type=self.option_type, k=self.k, strike=self.strike)
            # Check if model price is NaN (can happen from black_scholes)
        if np.isnan(model_price):
            # print("Model price calculation resulted in NaN")
            return 1e11 # Indicate error
        # print(f"Vol: {vol}, Model Price: {model_price}, Market Price: {self.market_price}")
        error = model_price - self.market_price
        # print(f"Objective Function Error: {error}")
        return error
    
    def objective_func_og(self, vol) -> float:
        # Return large value for invalid sigma to guide solver
        if vol <= 0:
            print("Invalid sigma value")
            return 1e10
        model_price = black_scholes_og(spot_price=self.spot_price, strike=self.strike, tte_years=self.tte_years, iv=vol, option_type=self.option_type, r=self.r)
            # Check if model price is NaN (can happen from black_scholes)
        if np.isnan(model_price):
            # print("Model price calculation resulted in NaN")
            return 1e11 # Indicate error
        # print(f"Vol: {vol}, Model Price: {model_price}, Market Price: {self.market_price}")
        error = model_price - self.market_price
        # print(f"Objective Function Error: {error}")
        return error

    def calculate_iv(self, low_vol, high_vol) -> float | None:
        """
        Calculate implied volatility using Brent's method
        """
        try:
            iv = brentq(self.objective_func, low_vol, high_vol, xtol=tol, rtol=np.float64(tol), full_output=False)
            if isinstance(iv, float):
                return iv
            else:
                # print("IV calculation did not return a float")
                return None
        except ValueError:
            # Could not find a root in the given interval
            # print("IV calculation failed: ValueError in root finding")
            return None
        
    def calculate_iv_og(self, low_vol, high_vol) -> float | None:
        """
        Calculate implied volatility using Brent's method
        """
        try:
            iv = brentq(self.objective_func_og, low_vol, high_vol, xtol=tol, rtol=np.float64(tol), full_output=False)
            if isinstance(iv, float):
                return iv
            else:
                # print("IV calculation did not return a float")
                return None
        except ValueError:
            # Could not find a root in the given interval
            # print("IV calculation failed: ValueError in root finding")
            return None

    def adjust_iv_limit(self, obj_low, obj_high):
        """
        Adjust the limits for implied volatility calculation
        """
        if abs(obj_low) < abs(obj_high): # Market price closer to low vol price
            high_vol_adj = self.high_vol * 3
            obj_high_adj = self.objective_func(high_vol_adj)
            if np.sign(obj_low) != np.sign(obj_high_adj):
                self.high_vol = high_vol_adj
            else:
                print("Could not adjust IV limits to bracket root")

        else: # Market price closer to high vol price
            low_vol_adj = self.low_vol * 0.2
            obj_low_adj = self.objective_func(low_vol_adj)
            if np.sign(obj_low_adj) != np.sign(obj_high):
                self.low_vol = low_vol_adj
            else:
                print("Could not adjust IV limits to bracket root")
    
    def iv_calculator(self) -> Optional[float]:
        """
        Main method to calculate implied volatility
        """
        # Calculate objective function values at initial vol limits
        obj_low = self.objective_func(self.low_vol)
        obj_high = self.objective_func(self.high_vol)

        # Calculate price at low and high vol for validation
        price_at_low_vol = black_scholes(spot_price=self.spot_price, tte_years=self.tte_years, iv=self.low_vol, option_type=self.option_type, k=self.k, strike=self.strike)
        price_at_high_vol = black_scholes(spot_price=self.spot_price, tte_years=self.tte_years, iv=self.high_vol, option_type=self.option_type, k=self.k, strike=self.strike)

        # Validate IV bounds
        if not valid_obj_func(obj_low, obj_high):
            print("Invalid objective function values for IV calculation")
            return None
        if not valid_price(self.market_price, price_at_low_vol, price_at_high_vol):
            self.adjust_iv_limit(obj_low, obj_high)
            adj_price_at_low_vol = black_scholes(spot_price=self.spot_price, tte_years=self.tte_years, iv=self.low_vol, option_type=self.option_type, k=self.k, strike=self.strike)
            adj_price_at_high_vol = black_scholes(spot_price=self.spot_price, tte_years=self.tte_years, iv=self.high_vol, option_type=self.option_type, k=self.k, strike=self.strike)
            if not valid_price(self.market_price, adj_price_at_low_vol, adj_price_at_high_vol):
                print(f"Market Price: {self.market_price}, Price at Low Vol: {adj_price_at_low_vol}, Price at High Vol: {adj_price_at_high_vol} for strike {self.strike}")
                print("Invalid price range for IV calculation")
                return None
        
        if np.sign(obj_low) == np.sign(obj_high):
            self.adjust_iv_limit(obj_low, obj_high) # Try adjusting limits
            # Recalculate objective function values after adjustment
            obj_low = self.objective_func(self.low_vol)
            obj_high = self.objective_func(self.high_vol)

        iv = self.calculate_iv(self.low_vol, self.high_vol)
    
        if not valid_iv(iv, self.low_vol, self.high_vol):
            print("Invalid implied volatility calculation")
            return None
        
        return iv

    def iv_calculator_og(self) -> Optional[float]:
        """
        Main method to calculate implied volatility using original BS formula
        """
        # Calculate objective function values at initial vol limits
        obj_low = self.objective_func_og(self.low_vol)
        obj_high = self.objective_func_og(self.high_vol)

        # Calculate price at low and high vol for validation
        price_at_low_vol = black_scholes_og(spot_price=self.spot_price, strike=self.strike, tte_years=self.tte_years, iv=self.low_vol, option_type=self.option_type, r=self.r)
        price_at_high_vol = black_scholes_og(spot_price=self.spot_price, strike=self.strike, tte_years=self.tte_years, iv=self.high_vol, option_type=self.option_type, r=self.r)

        # Validate IV bounds
        if not valid_obj_func(obj_low, obj_high):
            print("Invalid objective function values for IV calculation")
            return None
        if not valid_price(self.market_price, price_at_low_vol, price_at_high_vol):
            # self.adjust_iv_limit(obj_low, obj_high)
            # adj_price_at_low_vol = black_scholes_og(spot_price=self.spot_price, strike=self.strike, tte_years=self.tte_years, iv=self.low_vol, option_type=self.option_type, r=self.r)
            # adj_price_at_high_vol = black_scholes_og(spot_price=self.spot_price, strike=self.strike, tte_years=self.tte_years, iv=self.high_vol, option_type=self.option_type, r=self.r)
            # if not valid_price(self.market_price, adj_price_at_low_vol, adj_price_at_high_vol):
            #     print(f"Market Price: {self.market_price}, Price at Low Vol: {adj_price_at_low_vol}, Price at High Vol: {adj_price_at_high_vol} for strike {self.strike}")
            #     print("Invalid price range for IV calculation")
            return None
        
        if np.sign(obj_low) == np.sign(obj_high):
            self.adjust_iv_limit(obj_low, obj_high) # Try adjusting limits
            # Recalculate objective function values after adjustment
            obj_low = self.objective_func_og(self.low_vol)
            obj_high = self.objective_func_og(self.high_vol)

        iv = self.calculate_iv_og(self.low_vol, self.high_vol)
    
        if not valid_iv(iv, self.low_vol, self.high_vol):
            print("Invalid implied volatility calculation")
            return None
        
        return iv