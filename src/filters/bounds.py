from dataclasses import dataclass

@dataclass
class Bounds:
    """Class to hold bounds for filtering options"""
    low_vol_bound: float
    high_vol_bound: float