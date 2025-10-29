from scipy import interpolate
import numpy as np

def cubic_spline_interpolator(k: np.ndarray, iv: np.ndarray):
    """
    Interpolates implied volatilities using a cubic spline.
    """
    spline = interpolate.CubicSpline(k, iv, bc_type='natural')
    return spline