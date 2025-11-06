import pandas as pd
from src.core.interpolation_single_expiry.svi import SVI
from src.core.interpolation_single_expiry.svi_redo import raw_svi, svi_fit_direct
from src.orchestrators.df_creator import DFCreator
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

class SingleExpiry:
    def __init__(self, days_to_expiry: int, df: pd.DataFrame):
        self.single_df = df.loc[df["TTE_days"] == days_to_expiry].copy()
        
        # Validate that we have data
        if self.single_df.empty:
            available_days = sorted(df["TTE_days"].unique())
            raise ValueError(f"No data found for {days_to_expiry} days to expiry. "
                           f"Available days: {available_days}")
        
        print(f"Found {len(self.single_df)} options with {days_to_expiry} days to expiry")
        print(f"Moneyness range: {self.single_df['LogMoneyness'].min():.3f} to {self.single_df['LogMoneyness'].max():.3f}")

    def fit_svi(self) -> Tuple:
        try:

            params = svi_fit_direct(
                k=self.single_df["LogMoneyness"].to_numpy(), 
                w=self.single_df["W"].to_numpy(), 

                weights=self.single_df["Vega"].to_numpy(), 
                method="Nelder-Mead", 
                ngrid=5
            )
            return params.x
        except Exception as e:
            print(f"SVI fitting failed: {e}")
            raise

    def create_data_points(self, params: Tuple, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Create smooth curve points for plotting"""
        k_min = self.single_df["LogMoneyness"].min()
        k_max = self.single_df["LogMoneyness"].max()

        # Create smooth moneyness points
        k_values = np.linspace(k_min, k_max, num_points)

        # Calculate total variance using SVI formula
        w_values = raw_svi(par=params, k=k_values)

        # Convert to implied volatility: σ = √(w/T)
        tte_years = self.single_df["TTE_years"].iloc[0]
        implied_vol = np.sqrt(w_values / tte_years)  # Fixed formula
        # implied_vol = np.sqrt(w_values)

        return k_values, implied_vol

    def plot_svi(self):
        """Plot market data vs SVI fit"""
        try:
            params = self.fit_svi()
            k_smooth, iv_smooth = self.create_data_points(params)

            plt.figure(figsize=(12, 8))

            # Market data points (scattered)
            # plt.scatter(self.single_df["LogMoneyness"], self.single_df["CalcIV"],
            #            label="Market IV", color='blue', alpha=0.7, s=60, zorder=5)
            
            # plt.scatter(self.single_df["LogMoneyness"], self.single_df["IV"],
            #            label="Original Market IV", color='green', alpha=0.7, s=60, zorder=5)
            
            plt.scatter(self.single_df["LogMoneyness"], self.single_df["CalcIVOG"],
                       label="IV OG", color='orange', alpha=0.7, s=60, zorder=5)

            # SVI fitted curve (smooth line)
            plt.plot(k_smooth, iv_smooth, label="SVI Fit", color='red', linewidth=2, zorder=4)

            plt.xlabel("Log Moneyness")
            plt.ylabel("Implied Volatility")
            plt.title(f"SVI Fit to Market Data - {self.single_df['TTE_days'].iloc[0]} Days to Expiry")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Print fit parameters
            if len(params) == 5:
                a, b, rho, m, sigma = params
                plt.text(0.02, 0.98,
                         f'SVI Parameters:\na={a:.4f}, b={b:.4f}\nρ={rho:.4f}, m={m:.4f}, σ={sigma:.4f}',
                         transform=plt.gca().transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Plotting failed: {e}")
            raise


# class SingleExpiryFitting:
#     def __init__(self, days_to_expiry: int, df: pd.DataFrame):
#         self.single_df = df.loc[df["TTE_days"] == days_to_expiry].copy()
        
#         # Validate that we have data
#         if self.single_df.empty:
#             available_days = sorted(df["TTE_days"].unique())
#             raise ValueError(f"No data found for {days_to_expiry} days to expiry. "
#                            f"Available days: {available_days}")
        
#         print(f"Found {len(self.single_df)} options with {days_to_expiry} days to expiry")
#         print(f"Moneyness range: {self.single_df['LogMoneyness'].min():.3f} to {self.single_df['LogMoneyness'].max():.3f}")

#     def fit_svi(self) -> Tuple:
#         try:
#             svi_instance = SVI(
#                 k=self.single_df["LogMoneyness"].to_numpy(), 
#                 weights=self.single_df["Vega"].to_numpy(), 
#                 method="brute", 
#                 w=self.single_df["W"].to_numpy(), 
#                 tte_years=self.single_df["TTE_years"].iloc[0]
#             )
#             params = svi_instance.svi_fit_direct()
#             return params
#         except Exception as e:
#             print(f"SVI fitting failed: {e}")
#             raise
    
#     def create_data_points(self, params: Tuple, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
#         """Create smooth curve points for plotting"""
#         k_min = self.single_df["LogMoneyness"].min()
#         k_max = self.single_df["LogMoneyness"].max()
        
#         # Create smooth moneyness points
#         k_values = np.linspace(k_min, k_max, num_points)
        
#         # Calculate total variance using SVI formula
#         w_values = SVI.raw_svi(par=params, k=k_values)
        
#         # Convert to implied volatility: σ = √(w/T)
#         tte_years = self.single_df["TTE_years"].iloc[0]
#         implied_vol = np.sqrt(w_values / tte_years)  # Fixed formula
        
#         return k_values, implied_vol

#     def plot_svi(self):
#         """Plot market data vs SVI fit"""
#         try:
#             params = self.fit_svi()
#             k_smooth, iv_smooth = self.create_data_points(params)
            
#             plt.figure(figsize=(12, 8))
            
#             # Market data points (scattered)
#             plt.scatter(self.single_df["LogMoneyness"], self.single_df["IV"], 
#                        label="Market IV", color='blue', alpha=0.7, s=60, zorder=5)
            
#             # SVI fitted curve (smooth line)
#             plt.plot(k_smooth, iv_smooth, label="SVI Fit", color='red', linewidth=2, zorder=4)
            
#             plt.xlabel("Log Moneyness")
#             plt.ylabel("Implied Volatility")
#             plt.title(f"SVI Fit to Market Data - {self.single_df['TTE_days'].iloc[0]} Days to Expiry")
#             plt.legend()
#             plt.grid(True, alpha=0.3)
            
#             # Print fit parameters
#             if len(params) == 5:
#                 a, b, rho, m, sigma = params
#                 plt.text(0.02, 0.98, 
#                         f'SVI Parameters:\na={a:.4f}, b={b:.4f}\nρ={rho:.4f}, m={m:.4f}, σ={sigma:.4f}',
#                         transform=plt.gca().transAxes, verticalalignment='top',
#                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
#             plt.tight_layout()
#             plt.show()
            
#         except Exception as e:
#             print(f"Plotting failed: {e}")
#             raise


# def debug_data(df: pd.DataFrame):
#     """Debug function to check available data"""
#     print("DataFrame shape:", df.shape)
#     print("Columns:", list(df.columns))
#     print("\nAvailable TTE_days:", sorted(df["TTE_days"].unique()))
#     print("\nSample data:")
#     print(df[["TTE_days", "LogMoneyness", "IV", "W"]].head())
    
#     # Check for any missing values
#     print(f"\nMissing values:")
#     for col in ["TTE_days", "LogMoneyness", "IV", "W"]:
#         missing = df[col].isna().sum()
#         if missing > 0:
#             print(f"  {col}: {missing} missing values")


if __name__ == "__main__":    
    try:
        # Load data
        df_creator = DFCreator()
        data_df = df_creator.apply_greeks()
        
        # # Debug the data first
        # print("=== DATA DEBUG ===")
        # debug_data(data_df)
        
        # Use the first available expiry day
        available_days = sorted(data_df["TTE_days"].unique())
        
        if not available_days:
            print("No data available!")
        else:
            days_to_expiry = available_days[0]  # Use first available
            print(f"\n=== FITTING SVI FOR {days_to_expiry} DAYS ===")
            
            fitting = SingleExpiry(days_to_expiry=22, df=data_df)
            fitting.plot_svi()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()