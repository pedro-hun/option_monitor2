from scipy import interpolate
import numpy as np

def cubic_spline_interpolator(k, iv):
    """
    Interpolates implied volatilities using a cubic spline.
    Ensures k values are strictly increasing.
    """
    k = np.array(k)
    iv = np.array(iv)
    
    if len(k) != len(iv):
        raise ValueError("k and iv must have the same length")
    
    if len(k) < 2:
        raise ValueError("Need at least 2 points for interpolation")
    
    # Sort by k
    sorted_indices = np.argsort(k)
    k_sorted = k[sorted_indices]
    iv_sorted = iv[sorted_indices]

        # Remove duplicates, keeping first occurrence
    mask = np.concatenate([[True], k_sorted[1:] != k_sorted[:-1]])
    k_unique = k_sorted[mask]
    iv_unique = iv_sorted[mask]
    
    if len(k_unique) < 2:
        raise ValueError("Need at least 2 unique k values for interpolation")

    spline = interpolate.CubicSpline(k_unique, iv_unique, bc_type='natural')
    return spline