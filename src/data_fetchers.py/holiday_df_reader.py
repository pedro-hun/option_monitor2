from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Holidays:
    """Class to handle holiday data reading"""

    def read_holidays(self, holiday_path: str, date_column: str) -> np.ndarray:
        """
        Read holidays from an Excel file and return as a numpy array
        """
        # Load holidays from CSV
        holidays_df = pd.read_csv(holiday_path)

        # Get holidays in the correct format for numpy busday_count
        holidays_for_numpy = pd.to_datetime(holidays_df[date_column], format='%m/%d/%Y').values.astype('datetime64[D]')

        return holidays_for_numpy