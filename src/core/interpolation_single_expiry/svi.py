import numpy as np
from math import pi
import scipy.optimize as sop
from sklearn.metrics import mean_squared_error
from typing import Tuple

class SVI:
    def __init__(self, par: Tuple[float, float, float, float, float], k: np.ndarray, weights: np.ndarray, method: str, w: np.ndarray):
        self.par = par
        self.k = k
        self.weights = weights
        self.method = method
        self.w = w

    def raw_svi(self):
        """
        Returns total variance for a given set of parameters from RAW SVI
        parametrization at given moneyness points.
        @param par: Set of raw parameters, (a, b, rho, m, sigma)
        @type k: PdSeries
        @param k: Moneyness points to evaluate
        @return: Total variance
        """
        a, b, rho, m, sigma = self.par
        w = a + b * (rho * (self.k - m) + np.sqrt((self.k - m) ** 2 + sigma ** 2))
        return w


    def diff_svi(self):
        """
        First derivative of RAW SVI with respect to moneyness.
        @param par: Set of raw parameters, (a, b, rho, m, sigma)
        @type k: PdSeries
        @param k: Moneyness points to evaluate
        @return: First derivative evaluated at k points
        """
        _, b, rho, m, sigma = self.par
        return b*(rho+(self.k-m)/(np.sqrt((self.k-m)**2+sigma**2)))


    def diff2_svi(self):
        """
        Second derivative of RAW SVI with respect to moneyness.
        @param par: Set of raw parameters, (a, b, rho, m, sigma)
        @type k: PdSeries
        @param k: Moneyness points to evaluate
        @return: Second derivative evaluated at k points
        """
        _, b, _, m, sigma = self.par
        disc = (self.k-m)**2 + sigma**2
        return (b*sigma**2)/((disc)**(3/2))


    def gfun(self):
        """
        Computes the g(k) function. Auxiliary to retrieve implied density and
        essential to test for butterfly arbitrage.
        @param par: Set of raw parameters, (a, b, rho, m, sigma)
        @type k: PdSeries
        @param k: Moneyness points to evaluate
        @return: Function g(k) evaluated at k points
        """
        w = self.raw_svi()
        w1 = self.diff_svi()
        w2 = self.diff2_svi()

        g = (1-0.5*(self.k*w1/w))**2 - (0.25*w1**2)*(w**-1+0.25) + 0.5*w2
        return g


    def d1(self):
        """
        Auxiliary function to compute d1 from BSM model.
        @param par: Set of raw parameters, (a, b, rho, m, sigma)
        @type k: PdSeries
        @param k: Moneyness points to evaluate
        @return: Values of d1 evaluated at k points
        """
        v = np.sqrt(self.raw_svi())
        return -self.k/v + 0.5*v


    def d2(self):
        """
        Auxiliary function to compute d2 from BSM model.
        @param par: Set of raw parameters, (a, b, rho, m, sigma)
        @type k: PdSeries
        @param k: Moneyness points to evaluate
        @return: Values of d2 evaluated at k points
        """
        v = np.sqrt(self.raw_svi())
        return -self.k/v - 0.5*v


    def density(self):
        """
        Probability density implied by an SVI.
        @param par: Set of raw parameters, (a, b, rho, m, sigma)
        @type k: PdSeries
        @param k: Moneyness points to evaluate
        @return: Implied risk neutral probability density from an SVI
        """
        g = self.gfun()
        w = self.raw_svi()
        dtwo = self.d2()

        dens = (g / np.sqrt(2 * pi * w)) * np.exp(-0.5 * dtwo**2)
        return dens


    def svi_fit_direct(self, ngrid=5):
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
        bounds = [(0., max(self.w)),
                (0., None),
                (-1., 1.),
                (2*min(self.k), 2*max(self.k)),
                (0., None)]
        # Tuples for brute force
        bfranges = tuple(bounds)

        # Objective function using sklearn
        def obj_fun():
            return mean_squared_error(self.w, self.raw_svi(), sample_weight=self.weights)

        # Chooses algo and run
        if self.method == "brute":
            # Brute force algo. fmin will take place to refine solution
            p0 = sop.brute(obj_fun, bfranges, Ns=ngrid, full_output=True)
            return p0
        elif self.method == "DE":
            # Differential Evolution algo.
            p0 = sop.differential_evolution(obj_fun, bounds)
            return p0
        else:
            p0 = sop.minimize(obj_fun, x0=[np.min(self.w), max(2*(np.max(self.w)-np.max(self.w))/((np.max(self.k) - np.min(self.k)))), np.sign(self.w[-1] - self.w[0])*0.5, self.k[np.argmin(self.w)], max((np.max(self.k) - np.min(self.k))/ 4.0, 0.05)], method=self.method, bounds=bounds)
            return p0