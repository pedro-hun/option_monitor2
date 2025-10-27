from dataclasses import dataclass
import xlwings as xw
import pandas as pd

@dataclass
class ExcelReaderConfig:
    """Class to hold configuration for Excel reading"""
    excel_file: str
    sheet_name: str
    iv_col: str
    start_row: int
    end_row: int
    bid_col: str
    ask_col: str
    last_price_col: str
    strike_col: str
    open_interest_col: str
    volume_col: str
    ticker_col: str
    maturity_col: str
    spot_price_col: str


class ExcelReader:
    def __init__(self, config: ExcelReaderConfig):
        self.excel_file = config.excel_file
        self.sheet_name = config.sheet_name
        self.iv_col = config.iv_col
        self.bid_col = config.bid_col
        self.ask_col = config.ask_col
        self.last_price_col = config.last_price_col
        self.strike_col = config.strike_col
        self.open_interest_col = config.open_interest_col
        self.volume_col = config.volume_col
        self.ticker_col = config.ticker_col
        self.maturity_col = config.maturity_col
        self.start_row = config.start_row
        self.end_row = config.end_row
        self.spot_price_col = config.spot_price_col
        self.wb = None
        self.sht = None

    def read_sheet(self):
        """
        Open Excel workbook and return the specified sheet using instance variables.

        Returns:
        --------
        xlwings.Sheet
            The Excel sheet object
        """
        self.wb = xw.Book(self.excel_file)
        self.sht = self.wb.sheets[self.sheet_name]
        return self.sht

    def get_data(self):
        """
        Read moneyness and implied volatility data from Excel using instance variables.

        Returns:
        --------
        tuple
            (moneyness_array, iv_array, dataframe)
        """
        # Read sheet if not already open
        if self.sht is None or self.wb is None:
            self.read_sheet()
        
        # Ensure sheet is properly initialized
        if self.sht is None:
            raise RuntimeError("Failed to initialize Excel sheet")

        iv = self.sht.range(f"{self.iv_col}{self.start_row} : {self.iv_col}{self.end_row}").value
        bid = self.sht.range(f"{self.bid_col}{self.start_row} : {self.bid_col}{self.end_row}").value
        ask = self.sht.range(f"{self.ask_col}{self.start_row} : {self.ask_col}{self.end_row}").value
        last_price = self.sht.range(f"{self.last_price_col}{self.start_row} : {self.last_price_col}{self.end_row}").value
        strike = self.sht.range(f"{self.strike_col}{self.start_row} : {self.strike_col}{self.end_row}").value
        open_interest = self.sht.range(f"{self.open_interest_col}{self.start_row} : {self.open_interest_col}{self.end_row}").value
        volume = self.sht.range(f"{self.volume_col}{self.start_row} : {self.volume_col}{self.end_row}").value
        ticker = self.sht.range(f"{self.ticker_col}{self.start_row} : {self.ticker_col}{self.end_row}").value
        maturity = self.sht.range(f"{self.maturity_col}{self.start_row} : {self.maturity_col}{self.end_row}").value
        spot_price = self.sht.range(f"{self.spot_price_col}{self.start_row} : {self.spot_price_col}{self.end_row}").value
        


        data_df = pd.DataFrame({"bid": bid, "ask": ask, "lastPrice": last_price, "Strike": strike,
                                "openInterest": open_interest, "volume": volume,
                                "ticker": ticker, "Expiry": maturity, "IV": iv, "SpotPrice": spot_price})

        return data_df