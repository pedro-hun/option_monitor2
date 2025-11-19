import sys
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add the project root directory to Python path
project_root = r'c:\Users\pedro.hun\Documents\repos\option_monitor2'
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.core.entities.option import Option, OptionType
from src.core.entities.iv_calc import IVCalc
from src.core.interpolation_single_expiry.svi import SVI
from src.data_fetchers.excel_reader import ExcelReader, ExcelReaderConfig
from src.data_fetchers.holiday_df_reader import Holidays
from src.filters.option_filters import option_data_positive, intrinsic_lower_than_price, option_lower_than_underlying, valid_price, valid_obj_func, valid_iv
from src.utils.option_data_calc import get_type, calculate_tte_days, calculate_tte_years, calculate_mid_price, calculate_bid_ask_spread, calculate_relative_spread, calculate_forward_price, calculate_moneyness, calculate_log_moneyness, calculate_intrinsic_value
from src.core.entities.black_scholes import black_scholes, black_scholes_og, w, d1, d2, discount_factor
from src.core.greeks.greeks_calculation import vega, theta, rho, delta, gamma
from src.core.interpolation_single_expiry.svi_redo import  mi_svi, omega_svi, svi_fit_direct, jw_vt, jw_psi, jw_pt, jw_ct, jw_vhat, delta_svi, zeta_svi


holidays = Holidays()
excel_reader_config = ExcelReaderConfig(excel_file="C:/Users/pedro.hun/Documents/repos/option_monitor2/examples/data/petr_options_chain_4.xlsx", sheet_name="Sheet1", iv_col="I", start_row=2, end_row=1925, bid_col="E", ask_col="F", last_price_col="B", strike_col="C", open_interest_col="H", volume_col="D", ticker_col="A", maturity_col="G", spot_price_col="J")




class DataFrameCreator:
    def __init__(self):
        self.excel_reader = ExcelReader(excel_reader_config)

    def run(self) -> pd.DataFrame:
        
        holidays_df = holidays.read_holidays("C:/Users/pedro.hun/Documents/repos/option_monitor2/examples/data/feriados_nacionais.csv", "Data")
        data = self.excel_reader.get_data()
        data["OptionType"] = data["ticker"].apply(lambda x: get_type(x))
        data['TTE_days'] = data['Expiry'].apply(lambda row: calculate_tte_days(current_date=datetime(2025, 10, 21), expiry=row, holidays=holidays_df))
        data['TTE_years'] = data['TTE_days'].apply(lambda tte_days: calculate_tte_years(tte_days))
        data['MidPrice'] = data.apply(lambda row: calculate_mid_price(bid_price=row['bid'], ask_price=row['ask']), axis=1)
        data['Spread'] = data.apply(lambda row: calculate_bid_ask_spread(bid_price=row['bid'], ask_price=row['ask']), axis=1)
        data['RelativeSpread'] = data.apply(lambda row: calculate_relative_spread(mid_price=row['MidPrice'], spread=row['Spread']), axis=1)
        data['ForwardPrice'] = data.apply(lambda row: calculate_forward_price(spot_price=row['SpotPrice'], risk_free_rate=0.15, tte_years=row['TTE_years'], borrow=0), axis=1)
        data['Moneyness'] = data.apply(lambda row: calculate_moneyness(forward_price=row['ForwardPrice'], strike=row['Strike']), axis=1)
        data['LogMoneyness'] = data["Moneyness"].apply(lambda x: float(calculate_log_moneyness(moneyness=x)))
        data["IntrinsicValue"] = data.apply(lambda row: calculate_intrinsic_value(spot_price=row['SpotPrice'], strike=row['Strike'], option_type=row['OptionType']), axis=1)
        
        return data

class BasicFilters:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def apply_filters(self) -> pd.DataFrame:
        self.df["mask_data_positive"] = self.df.apply(lambda row: option_data_positive(bid=row["bid"], ask=row["ask"], tte_years=row["TTE_years"], iv=row["IV"]), axis=1)
        self.df["mask_intrinsic"] = self.df.apply(lambda row: intrinsic_lower_than_price(intrinsic_value=row["IntrinsicValue"], mid_price=row["MidPrice"]), axis=1)
        self.df["mask_underlying"] = self.df.apply(lambda row: option_lower_than_underlying(mid=row["MidPrice"], spot_price=row["SpotPrice"], strike=row["Strike"], tte_years=row["TTE_years"], option_type=row["OptionType"]), axis=1)
        
        filtered_df = self.df[self.df["mask_data_positive"] & self.df["mask_intrinsic"] & self.df["mask_underlying"]].drop(columns=["mask_data_positive", "mask_intrinsic", "mask_underlying"]).copy()
        return filtered_df
        # return self.df
    
class GreeksCalc:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_greeks(self) -> pd.DataFrame:
        # Placeholder for greeks calculation logic
        self.df["Vega"] = self.df.apply(lambda row: vega(spot_price=row["SpotPrice"], tte_years=row["TTE_years"], d1=row["D1"], borrow=0), axis=1)
        self.df["Theta"] = self.df.apply(lambda row: theta(spot_price=row["SpotPrice"], strike=row["Strike"], tte_years=row["TTE_years"], sigma=row["CalcIVOG"], d1=row["D1"], d2=row["D2"], r=0.15, option_type=row["OptionType"], borrow=0), axis=1)
        self.df["Rho"] = self.df.apply(lambda row: rho(strike=row["Strike"], tte_years=row["TTE_years"], d2=row["D2"], r=0.15, option_type=row["OptionType"]), axis=1)
        self.df["Delta"] = self.df.apply(lambda row: delta(tte_years=row["TTE_years"], d1=row["D1"], option_type=row["OptionType"], borrow=0), axis=1)
        self.df["Gamma"] = self.df.apply(lambda row: gamma(spot_price=row["SpotPrice"], tte_years=row["TTE_years"], sigma=row["CalcIVOG"], d1=row["D1"], borrow=0), axis=1)
        return self.df
    
class DFCreator:
    def __init__(self):
        self.dataframe_creator = DataFrameCreator()
        self.filtered_df = pd.DataFrame()

    def create_filtered_dataframe(self):
        df = self.dataframe_creator.run()
        basic_filters = BasicFilters(df)
        self.filtered_df = basic_filters.apply_filters()
        return self.filtered_df

    
    # def apply_iv_calc(self):
    #     self.filtered_df = self.create_filtered_dataframe().copy()
    #     self.filtered_df["CalcIV"] = self.filtered_df.apply(lambda row: IVCalc(
    #         market_price=row["MidPrice"],
    #         forward_price=row["ForwardPrice"],
    #         spot_price=row["SpotPrice"],
    #         strike=row["Strike"],
    #         tte_years=row["TTE_years"],
    #         option_type=row["OptionType"],
    #         k=row["LogMoneyness"],
    #         r=0.15,
    #         sigma=row["IV"]
    #     ).iv_calculator(), axis=1)
    #     return self.filtered_df

    def apply_iv_calc_og(self):
        self.filtered_df = self.create_filtered_dataframe().copy()
        self.filtered_df["CalcIVOG"] = self.filtered_df.apply(lambda row: IVCalc(
            market_price=row["MidPrice"],
            forward_price=row["ForwardPrice"],
            spot_price=row["SpotPrice"],
            strike=row["Strike"],
            tte_years=row["TTE_years"],
            option_type=row["OptionType"],
            k=row["LogMoneyness"],
            r=0.15,
            sigma=row["IV"]
        ).iv_calculator_og(), axis=1)
        return self.filtered_df

    def apply_bs(self):
        self.filtered_df = self.apply_iv_calc_og().copy()
        self.filtered_df["W"] = self.filtered_df.apply(lambda row: w(iv=row["CalcIVOG"], tte_years=row["TTE_years"]), axis=1)
        self.filtered_df["D1"] = self.filtered_df.apply(lambda row: d1(w=row["W"], k=row["LogMoneyness"]), axis=1)
        self.filtered_df["D2"] = self.filtered_df.apply(lambda row: d2(w=row["W"], k=row["LogMoneyness"]), axis=1)
        self.filtered_df["BSPrice"] = self.filtered_df.apply(lambda row: black_scholes(
            spot_price=row["SpotPrice"],
            tte_years=row["TTE_years"],
            iv=row["IV"],
            option_type=row["OptionType"],
            k=row["LogMoneyness"],
            strike=row["Strike"]
        ), axis=1)
        self.filtered_df["BSPriceOG"] = self.filtered_df.apply(lambda row: black_scholes_og(
            spot_price=row["SpotPrice"],
            strike=row["Strike"],
            tte_years=row["TTE_years"],
            iv=row["IV"],
            option_type=row["OptionType"],
            r=0.15
        ), axis=1)

        return self.filtered_df
    
    def apply_greeks(self):
        self.filtered_df = self.apply_bs().copy()
        greeks_calculator = GreeksCalc(self.filtered_df)
        self.filtered_df = greeks_calculator.calculate_greeks()
        return self.filtered_df
    
    
    
    def calc_params(self) -> pd.DataFrame:
        """Calculate SVI parameters for each expiry and add them to the dataframe"""
        
        self.filtered_df = self.apply_greeks().copy().dropna()
        
        
        # Initialize parameter columns
        self.filtered_df["svi_a"] = np.nan
        self.filtered_df["svi_b"] = np.nan
        self.filtered_df["svi_rho"] = np.nan
        self.filtered_df["svi_m"] = np.nan
        self.filtered_df["svi_sigma"] = np.nan
        
        for tte_days in self.filtered_df['TTE_days'].unique():
            try:
                print(f"Fitting SVI for TTE_days = {tte_days}")
                if len(self.filtered_df[self.filtered_df["TTE_days"] == tte_days]) < 5:
                    print(f"Skipping TTE_days = {tte_days} due to insufficient data points")
                    continue
                
                # Filter data for this expiry
                single_expiry_df = self.filtered_df.loc[self.filtered_df["TTE_days"] == tte_days].copy()
                
                
                # Prepare data for SVI fitting
                k_values = single_expiry_df["LogMoneyness"].to_numpy()
                w_values = single_expiry_df["W"].to_numpy()
                weights = single_expiry_df["Vega"].to_numpy()

                # Debug the input data
                print(f"k_values range: [{k_values.min():.4f}, {k_values.max():.4f}]")
                print(f"w_values range: [{w_values.min():.4f}, {w_values.max():.4f}]")
                print(f"weights range: [{weights.min():.4f}, {weights.max():.4f}]")            
                
                # Fit SVI using svi_fit_direct from svi_redo.py
                result = svi_fit_direct(
                    k=k_values,
                    w=w_values,
                    weights=weights,
                    method="COBYQA",
                    ngrid=5
                )
                
                params = result.x
                    
                # Assign parameters to all rows with this expiry
                self.filtered_df.loc[self.filtered_df["TTE_days"] == tte_days, "svi_a"] = params[0]
                self.filtered_df.loc[self.filtered_df["TTE_days"] == tte_days, "svi_b"] = params[1]
                self.filtered_df.loc[self.filtered_df["TTE_days"] == tte_days, "svi_rho"] = params[2]
                self.filtered_df.loc[self.filtered_df["TTE_days"] == tte_days, "svi_m"] = params[3]
                self.filtered_df.loc[self.filtered_df["TTE_days"] == tte_days, "svi_sigma"] = params[4]
            except Exception as e:
                print(f"Error fitting SVI for TTE_days = {tte_days}: {e}")
                continue
        return self.filtered_df
    
    def apply_jw(self) -> pd.DataFrame:
        self.filtered_df = self.calc_params().copy()
        self.filtered_df["vt"] = self.filtered_df.apply(lambda row: jw_vt(a=row["svi_a"], b=row["svi_b"], rho=row["svi_rho"], m=row["svi_m"], sigma=row["svi_sigma"], tte_years=row["TTE_years"]), axis=1)
        self.filtered_df["psi"] = self.filtered_df.apply(lambda row: jw_psi(a=row["svi_a"], b=row["svi_b"], rho=row["svi_rho"], m=row["svi_m"], sigma=row["svi_sigma"], tte_years=row["TTE_years"]), axis=1)
        self.filtered_df["p_t"] = self.filtered_df.apply(lambda row: jw_pt(a=row["svi_a"], b=row["svi_b"], rho=row["svi_rho"], m=row["svi_m"], sigma=row["svi_sigma"], tte_years=row["TTE_years"]), axis=1)
        self.filtered_df["c_t"] = self.filtered_df.apply(lambda row: jw_ct(a=row["svi_a"], b=row["svi_b"], rho=row["svi_rho"], m=row["svi_m"], sigma=row["svi_sigma"], tte_years=row["TTE_years"]), axis=1)
        self.filtered_df["vhat"] = self.filtered_df.apply(lambda row: jw_vhat(a=row["svi_a"], b=row["svi_b"], rho=row["svi_rho"], m=row["svi_m"], sigma=row["svi_sigma"], tte_years=row["TTE_years"]), axis=1)
        return self.filtered_df
    
    def apply_natural(self) -> pd.DataFrame:
        self.filtered_df = self.apply_jw().copy()
        self.filtered_df["svi_delta"] = self.filtered_df.apply(lambda row: delta_svi(a=row["svi_a"], b=row["svi_b"], rho=row["svi_rho"], m=row["svi_m"], sigma=row["svi_sigma"]), axis=1)
        self.filtered_df["svi_mi"] = self.filtered_df.apply(lambda row: mi_svi(a=row["svi_a"], b=row["svi_b"], rho=row["svi_rho"], m=row["svi_m"], sigma=row["svi_sigma"]), axis=1)
        self.filtered_df["svi_omega"] = self.filtered_df.apply(lambda row: omega_svi(a=row["svi_a"], b=row["svi_b"], rho=row["svi_rho"], m=row["svi_m"], sigma=row["svi_sigma"]), axis=1)
        self.filtered_df["svi_zeta"] = self.filtered_df.apply(lambda row: zeta_svi(a=row["svi_a"], b=row["svi_b"], rho=row["svi_rho"], m=row["svi_m"], sigma=row["svi_sigma"]), axis=1)
        return self.filtered_df





