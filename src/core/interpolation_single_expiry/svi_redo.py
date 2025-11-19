import numpy as np
from math import pi
import scipy.optimize as sop
from sklearn.metrics import mean_squared_error
from typing import Tuple


def raw_svi(par, k):
    """
    Returns total variance for a given set of parameters from RAW SVI
    parametrization at given moneyness points.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Total variance
    """
    a, b, rho, m, sigma = par
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
    return w

def jw_par(a, b, rho, m, sigma, tte_years):
    """
    Returns a set of parameters from JW SVI parametrization at given moneyness points.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Total variance
    """
    vt = (a + b * (- rho * m + np.sqrt(m ** 2 + sigma ** 2)))/tte_years
    psi = 0.5 * b * (- m / np.sqrt(m ** 2 + sigma ** 2) + rho)/np.sqrt(vt*tte_years)
    pt = b * (1 - rho)/np.sqrt(vt*tte_years)
    ct = b * (1 + rho)/np.sqrt(vt*tte_years)
    vhat = (a + b * (sigma * np.sqrt(1 - rho**2)))/tte_years
    jw_par = vt, psi, pt, ct, vhat
    return jw_par

def jw_vt(a, b, rho, m, sigma, tte_years):
    """
    Returns vt from JW SVI parametrization at given moneyness points.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Total variance
    """
    vt = (a + b * (- rho * m + np.sqrt(m ** 2 + sigma ** 2)))/tte_years
    return vt

def jw_psi(a, b, rho, m, sigma, tte_years):
    """
    Returns psi from JW SVI parametrization at given moneyness points.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Total variance
    """
    vt = (a + b * (- rho * m + np.sqrt(m ** 2 + sigma ** 2)))/tte_years
    psi = 0.5 * b * (- m / np.sqrt(m ** 2 + sigma ** 2) + rho)/np.sqrt(vt*tte_years)
    return psi

def jw_pt(a, b, rho, m, sigma, tte_years):
    """
    Returns p_t from JW SVI parametrization at given moneyness points.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Total variance
    """
    vt = (a + b * (- rho * m + np.sqrt(m ** 2 + sigma ** 2)))/tte_years
    pt = b * (1 - rho)/np.sqrt(vt*tte_years)
    return pt

def jw_ct(a, b, rho, m, sigma, tte_years):
    """
    Returns c_t from JW SVI parametrization at given moneyness points.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Total variance
    """
    vt = (a + b * (- rho * m + np.sqrt(m ** 2 + sigma ** 2)))/tte_years
    ct = b * (1 + rho)/np.sqrt(vt*tte_years)
    return ct

def jw_vhat(a, b, rho, m, sigma, tte_years):
    """
    Returns vhat from JW SVI parametrization at given moneyness points.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Total variance
    """
    vhat = (a + b * (sigma * np.sqrt(1 - rho**2)))/tte_years
    return vhat

def delta_svi(a, b, rho, m, sigma):
    """
    Delta from the natural SVI parametrization.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Delta evaluated at k points
    """
    omega_svi = 2*b*sigma/ (1 - rho**2)
    return a - 0.5*omega_svi*(1 - rho**2)

def mi_svi(a, b, rho, m, sigma):
    """
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Minimum implied variance evaluated at k points
    """
    return m + rho*sigma/ np.sqrt(1 - rho**2)

def omega_svi(a, b, rho, m, sigma):
    """
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Omega parameter evaluated at k points
    """
    return 2*b*sigma/ (1 - rho**2)

def zeta_svi(a, b, rho, m, sigma):

    return (1 - rho**2)/ (sigma)

def diff_svi(par, k):
    """
    First derivative of RAW SVI with respect to moneyness.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: First derivative evaluated at k points
    """
    _, b, rho, m, sigma = par
    return b*(rho+(k-m)/(np.sqrt((k-m)**2+sigma**2)))


def diff2_svi(par, k):
    """
    Second derivative of RAW SVI with respect to moneyness.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Second derivative evaluated at k points
    """
    _, b, _, m, sigma = par
    disc = (k-m)**2 + sigma**2
    return (b*sigma**2)/((disc)**(3/2))


def gfun(par, k):
    """
    Computes the g(k) function. Auxiliary to retrieve implied density and
    essential to test for butterfly arbitrage.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Function g(k) evaluated at k points
    """
    w = raw_svi(par, k)
    w1 = diff_svi(par, k)
    w2 = diff2_svi(par, k)

    g = (1-0.5*(k*w1/w))**2 - (0.25*w1**2)*(w**-1+0.25) + 0.5*w2
    return g


def d1(par, k):
    """
    Auxiliary function to compute d1 from BSM model.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Values of d1 evaluated at k points
    """
    v = np.sqrt(raw_svi(par, k))
    return -k/v + 0.5*v


def d2(par, k):
    """
    Auxiliary function to compute d2 from BSM model.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Values of d2 evaluated at k points
    """
    v = np.sqrt(raw_svi(par, k))
    return -k/v - 0.5*v

def density(par, k):
    """
    Probability density implied by an SVI.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Implied risk neutral probability density from an SVI
    """
    g = gfun(par, k)
    w = raw_svi(par, k)
    dtwo = d2(par, k)

    dens = (g / np.sqrt(2 * pi * w)) * np.exp(-0.5 * dtwo**2)
    return dens



def svi_fit_direct(k, w, weights, method, ngrid=5):
    """
    Direct procedure to calibrate a RAW SVI
    :param k: Moneyness
    :param w: Market total variance
    :param weights: Weights. Do not need to sum to 1
    :param method: Method of optimization. Currently implemented: "brute", "DE"
    :param ngrid: Number of grid points for each parameter in brute force
    algorithm
    :return: SVI parameters (a, b, rho, m, sigma)
    """
    # Bounds on parameters
    bounds = [(1e-6, max(w)),
               (1e-6, 2.),
               (-0.999, 0.999),
               (min(k) * 1.5 if min(k) < 0 else min(k) * 0.5, max(k) * 1.5 if max(k) > 0 else max(k) * 0.5),
               (1e-6, 2.)]
    # Tuples for brute force
    bfranges = tuple(bounds)

    # Objective function
    def obj_fun(par):
        # model_w = raw_svi(par, k)
        # normalized_weights = weights / np.sum(weights)
        # normalized_weights = normalized_weights/np.sum(normalized_weights)
        # error = np.sum(normalized_weights * (model_w - w)**2)
        # return error
        return mean_squared_error(w, raw_svi(par, k), sample_weight=weights)
        
    initial_guess = [np.min(w),
                    max((np.max(w)-np.min(w))*2/((np.max(k) - np.min(k))), 1e-3),
                    np.clip(np.sign(w[-1] - w[0])*0.5, -0.95, 0.95),
                    k[np.argmin(w)],
                    max((np.max(k) - np.min(k))/ 4.0, 0.05)]
    
    print(f"Initial guess for SVI parameters: {initial_guess}")

    # Chooses algo and run
    if method == "brute":
        # Brute force algo. fmin will take place to refine solution
        p0 = sop.brute(obj_fun, bfranges, Ns=ngrid, full_output=True)
        return p0
    elif method == "DE":
        # Differential Evolution algo.
        p0 = sop.differential_evolution(obj_fun, bounds)
        if not p0.success:
            print(f"Warning: DE Optimization failed: {p0.message}")
        return p0
    else:
        p0 = sop.minimize(obj_fun, x0=initial_guess, method=method, bounds=bounds, options={'maxiter':10000})
        if not p0.success:
            print(f"Warning: Minimize Optimization failed: {p0.message}")
        return p0