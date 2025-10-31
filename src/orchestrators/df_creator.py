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
from src.core.entities.black_scholes import black_scholes


holidays = Holidays()
excel_reader_config = ExcelReaderConfig(excel_file="C:/Users/pedro.hun/Documents/repos/option_monitor2/examples/data/petr_options_chain_3.xlsx", sheet_name="Sheet1", iv_col="I", start_row=2, end_row=1925, bid_col="E", ask_col="F", last_price_col="B", strike_col="C", open_interest_col="H", volume_col="D", ticker_col="A", maturity_col="G", spot_price_col="J")




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
        data["IntrinsicValue"] = data.apply(lambda row: calculate_intrinsic_value(forward_price=row['ForwardPrice'], strike=row['Strike'], option_type=row['OptionType']), axis=1)
        
        return data

class BasicFilters:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def apply_filters(self) -> pd.DataFrame:
        self.df["mask1"] = self.df.apply(lambda row: option_data_positive(bid=row["bid"], ask=row["ask"], tte_years=row["TTE_years"], iv=row["IV"]), axis=1)
        self.df["mask2"] = self.df.apply(lambda row: intrinsic_lower_than_price(intrinsic_value=row["IntrinsicValue"], mid_price=row["MidPrice"]), axis=1)
        self.df["mask3"] = self.df.apply(lambda row: option_lower_than_underlying(mid=row["MidPrice"], spot_price=row["SpotPrice"], option_type=row["OptionType"]), axis=1)
        
        filtered_df = self.df[self.df["mask1"] & self.df["mask2"] & self.df["mask3"]].drop(columns=["mask1", "mask2", "mask3"]).copy()

        return filtered_df
    
class DFCreator:
    def __init__(self):
        self.dataframe_creator = DataFrameCreator()
        self.filtered_df = pd.DataFrame()

    def create_filtered_dataframe(self):
        df = self.dataframe_creator.run()
        basic_filters = BasicFilters(df)
        self.filtered_df = basic_filters.apply_filters()
        return self.filtered_df

    def apply_bs(self):
        self.filtered_df = self.create_filtered_dataframe().copy()
        self.filtered_df["BSPrice"] = self.filtered_df.apply(lambda row: black_scholes(
            spot_price=row["SpotPrice"],
            strike=row["Strike"],
            tte_years=row["TTE_years"],
            iv=row["IV"],
            r=0.15,
            option_type=row["OptionType"],
            k=row["LogMoneyness"],
            borrow=0
        ), axis=1)
        return self.filtered_df

    
    def apply_iv_calc(self):
        self.filtered_df = self.apply_bs().copy()
        self.filtered_df["CalcIV"] = self.filtered_df.apply(lambda row: IVCalc(
            market_price=row["MidPrice"],
            forward_price=row["ForwardPrice"],
            spot_price=row["SpotPrice"],
            strike=row["Strike"],
            tte_years=row["TTE_years"],
            option_type=row["OptionType"],
            k=row["LogMoneyness"],
            r=0.15,
            sigma=row["IV"]
        ).iv_calculator(), axis=1)
        return self.filtered_df
    
if __name__ == "__main__":
    df_creator = DFCreator()
    final_df = df_creator.apply_iv_calc()
    print(final_df)


    

