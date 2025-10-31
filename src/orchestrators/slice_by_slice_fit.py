from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.core.entities.option import Option, OptionType
from src.core.entities.iv_calc import IVCalc
from src.core.interpolation_single_expiry.svi import SVI
from src.data_fetchers.excel_reader import ExcelReader, ExcelReaderConfig
from src.filters.option_filters import option_data_positive, intrinsic_lower_than_price, option_lower_than_underlying, valid_price, valid_obj_func, valid_iv
from src.utils.option_data_calc import calculate_tte_days, calculate_tte_years, calculate_mid_price, calculate_bid_ask_spread, calculate_relative_spread, calculate_forward_price, calculate_moneyness, calculate_log_moneyness, calculate_intrinsic_value

logger = logging.getLogger(__name__)

class SliceBySliceSVIOrchestrator:
    """
    Orchestrates the slice-by-slice SVI surface fitting process
    """
    
    def __init__(self, 
                 data_loader: DataLoader,
                 min_moneyness: float = 0.8,
                 max_moneyness: float = 1.2,
                 min_tte_days: int = 7,
                 max_tte_days: int = 365):
        self.data_loader = data_loader
        self.min_moneyness = min_moneyness
        self.max_moneyness = max_moneyness
        self.min_tte_days = min_tte_days
        self.max_tte_days = max_tte_days
        self.svi_surface: Optional[SVISurface] = None
        
    def load_and_process_options(self, underlying_symbol: str, 
                               current_date: Optional[datetime] = None) -> List[Option]:
        """
        Load option data and convert to Option objects
        """
        logger.info(f"Loading options data for {underlying_symbol}")
        
        # Load raw option data
        raw_data = self.data_loader.load_options_data(underlying_symbol)
        
        # Convert to Option objects
        options = []
        for _, row in raw_data.iterrows():
            try:
                option = Option(
                    strike=float(row['strike']),
                    expiry=pd.to_datetime(row['expiry']),
                    option_type=OptionType.CALL if row['option_type'].lower() == 'call' else OptionType.PUT,
                    underlying=row['underlying'],
                    spot_price=float(row['spot_price']),
                    bid_price=float(row['bid_price']) if pd.notna(row['bid_price']) else None,
                    ask_price=float(row['ask_price']) if pd.notna(row['ask_price']) else None,
                    last_price=float(row['last_price']) if pd.notna(row['last_price']) else None,
                    volume=int(row['volume']) if pd.notna(row['volume']) else None,
                    open_interest=int(row['open_interest']) if pd.notna(row['open_interest']) else None
                )
                options.append(option)
            except Exception as e:
                logger.warning(f"Failed to process option row: {e}")
                continue
                
        logger.info(f"Loaded {len(options)} options")
        return options
    
    def calculate_implied_volatilities(self, options: List[Option], 
                                     risk_free_rate: float) -> Dict[str, List[Tuple[Option, float]]]:
        """
        Calculate implied volatilities for all options and group by expiry
        """
        logger.info("Calculating implied volatilities")
        
        iv_by_expiry = {}
        
        for option in options:
            try:
                # Calculate time to expiry in years
                tte_days = option.calculate_tte_business_days()
                if tte_days < self.min_tte_days or tte_days > self.max_tte_days:
                    continue
                    
                tte_years = tte_days / 365.0
                
                # Use mid price for IV calculation
                if option.bid_price and option.ask_price:
                    market_price = (option.bid_price + option.ask_price) / 2
                elif option.last_price:
                    market_price = option.last_price
                else:
                    continue
                
                # Calculate forward price (simplified - assuming no dividends)
                forward_price = option.spot_price * np.exp(risk_free_rate * tte_years)
                
                # Calculate moneyness filter
                moneyness = option.strike / option.spot_price
                if moneyness < self.min_moneyness or moneyness > self.max_moneyness:
                    continue
                
                # Calculate IV
                iv_calc = IVCalc(
                    market_price=market_price,
                    forward_price=forward_price,
                    strike=option.strike,
                    tte_years=tte_years,
                    r=risk_free_rate,
                    option_type=option.option_type.value,
                    k=np.log(moneyness)  # log moneyness
                )
                
                iv = iv_calc.iv_calculator()
                
                if iv is not None and iv > 0:
                    expiry_key = option.expiry.strftime('%Y-%m-%d')
                    if expiry_key not in iv_by_expiry:
                        iv_by_expiry[expiry_key] = []
                    iv_by_expiry[expiry_key].append((option, iv))
                    
            except Exception as e:
                logger.warning(f"Failed to calculate IV for option {option.strike}: {e}")
                continue
        
        logger.info(f"Calculated IVs for {sum(len(v) for v in iv_by_expiry.values())} options across {len(iv_by_expiry)} expiries")
        return iv_by_expiry
    
    def fit_svi_slices(self, iv_by_expiry: Dict[str, List[Tuple[Option, float]]], 
                      risk_free_rate: float) -> Dict[str, SVIFit]:
        """
        Fit SVI model to each expiry slice
        """
        logger.info("Fitting SVI slices")
        
        svi_fits = {}
        
        for expiry_key, option_iv_pairs in iv_by_expiry.items():
            try:
                if len(option_iv_pairs) < 5:  # Need minimum options for fitting
                    logger.warning(f"Skipping {expiry_key}: insufficient data points ({len(option_iv_pairs)})")
                    continue
                
                # Extract data for fitting
                log_moneyness = []
                total_variances = []
                spot_price = option_iv_pairs[0][0].spot_price  # Same for all options
                
                for option, iv in option_iv_pairs:
                    tte_days = option.calculate_tte_business_days()
                    tte_years = tte_days / 365.0
                    
                    k = np.log(option.strike / spot_price)  # log moneyness
                    w = iv * iv * tte_years  # total variance
                    
                    log_moneyness.append(k)
                    total_variances.append(w)
                
                # Convert to numpy arrays
                k_array = np.array(log_moneyness)
                w_array = np.array(total_variances)
                
                # Fit SVI model
                svi_fit = SVIFit(k_array, w_array)
                success = svi_fit.fit()
                
                if success:
                    svi_fits[expiry_key] = svi_fit
                    logger.info(f"Successfully fitted SVI for {expiry_key}")
                else:
                    logger.warning(f"Failed to fit SVI for {expiry_key}")
                    
            except Exception as e:
                logger.error(f"Error fitting SVI for {expiry_key}: {e}")
                continue
        
        logger.info(f"Successfully fitted SVI for {len(svi_fits)} expiries")
        return svi_fits
    
    def create_svi_surface(self, svi_fits: Dict[str, SVIFit], 
                          spot_price: float) -> SVISurface:
        """
        Create SVI surface from individual slice fits
        """
        logger.info("Creating SVI surface")
        
        expiries = []
        parameters = []
        
        for expiry_key, svi_fit in svi_fits.items():
            expiry_date = pd.to_datetime(expiry_key)
            expiries.append(expiry_date)
            parameters.append(svi_fit.get_parameters())
        
        # Create surface
        surface = SVISurface(expiries, parameters, spot_price)
        
        logger.info(f"Created SVI surface with {len(expiries)} expiry slices")
        return surface
    
    def orchestrate_svi_fitting(self, underlying_symbol: str, 
                               risk_free_rate: float = 0.05,
                               current_date: Optional[datetime] = None) -> SVISurface:
        """
        Main orchestration method to create SVI surface
        """
        logger.info(f"Starting SVI surface fitting for {underlying_symbol}")
        
        try:
            # Step 1: Load and process options
            options = self.load_and_process_options(underlying_symbol, current_date)
            
            # Step 2: Filter options for fitting
            filtered_options = filter_options_for_fitting(options)
            
            # Step 3: Calculate implied volatilities
            iv_by_expiry = self.calculate_implied_volatilities(filtered_options, risk_free_rate)
            
            if not iv_by_expiry:
                raise ValueError("No valid implied volatilities calculated")
            
            # Step 4: Fit SVI to each expiry slice
            svi_fits = self.fit_svi_slices(iv_by_expiry, risk_free_rate)
            
            if not svi_fits:
                raise ValueError("No successful SVI fits")
            
            # Step 5: Create surface
            spot_price = options[0].spot_price if options else 100.0
            self.svi_surface = self.create_svi_surface(svi_fits, spot_price)
            
            logger.info("SVI surface fitting completed successfully")
            return self.svi_surface
            
        except Exception as e:
            logger.error(f"SVI surface fitting failed: {e}")
            raise
    
    def get_surface(self) -> Optional[SVISurface]:
        """Get the fitted SVI surface"""
        return self.svi_surface

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize orchestrator
    data_loader = DataLoader()  # Assumes you have this implemented
    orchestrator = SliceBySliceSVIOrchestrator(data_loader)
    
    # Fit SVI surface
    try:
        surface = orchestrator.orchestrate_svi_fitting("PETR4", risk_free_rate=0.1175)
        print(f"SVI surface created with {len(surface.expiries)} expiries")
    except Exception as e:
        print(f"Failed to create SVI surface: {e}")