from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import math
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CalibrationError(RuntimeError):
    """Raised when the calibration procedure fails to converge."""


class InvalidSliceError(ValueError):
    """Raised when an observed slice violates basic viability assumptions."""


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedSSVISlice:
    """
    Immutable snapshot of slice-level statistics extracted from option data.

    Attributes
    ----------
    expiry : pd.Timestamp
        Expiry date or comparable identifier for the slice.
    time_to_expiry : float
        Time to expiry in years.
    theta : float
        ATM total variance (sigma_atm^2 * T) for the slice.
    psi : float
        Sum of left/right wing slopes (theta * phi(theta)).
    p : float
        Left-wing slope (Put wing).
    c : float
        Right-wing slope (Call wing).
    svi_rho : float
        Raw skew (assumed to come from prior SVI fits), used for diagnostics.
    metadata : Dict[str, float]
        Optional container with additional metrics (e.g., bid-ask spread stats).
    """

    expiry: pd.Timestamp
    time_to_expiry: float
    theta: float
    psi: float
    p: float
    c: float
    svi_rho: float
    metadata: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Ensure the slice is numerically viable for SSVI calibration.
        """
        if self.time_to_expiry <= 0:
            raise InvalidSliceError("time_to_expiry must be positive.")
        if self.theta <= 0:
            raise InvalidSliceError("theta must be positive.")
        # if any(val < 0 for val in (self.psi, self.p, self.c)):
        #     print(f"Invalid values detected: psi={self.psi}, p={self.p}, c={self.c}, for expiry={self.expiry}")
        #     raise InvalidSliceError("psi, p, and c must be non-negative.")
        if -self.p > self.psi * 2.0:
            print(f"Invalid psi detected: psi={self.psi}, p={self.p}, c={self.c}, for expiry={self.expiry}")
            raise InvalidSliceError("psi must be at least 2 * -p.")
        if 2.0 * self.psi > self.c:
            print(f"Invalid psi detected: psi={self.psi}, p={self.p}, c={self.c}, for expiry={self.expiry}")
            raise InvalidSliceError("psi must be at most c / 2.")

    @property
    def tau(self) -> float:
        """Convenience alias for time-to-expiry."""
        return self.time_to_expiry

    @property
    def observed_rho(self) -> float:
        """
        Compute empirical rho implied by the left/right wing slopes.
        """
        denominator = self.c + self.p
        if denominator == 0:
            return float("nan")
        return (self.c - self.p) / denominator


@dataclass(frozen=True)
class SSVIParameters:
    """
    Global SSVI hyper-parameters.

    Attributes
    ----------
    rho : float
        Correlation parameter, must lie strictly inside (-1, 1) to avoid arbitrage.
    eta : float
        Positive scaling factor for the slope function.
    gamma : float
        Curvature exponent, typically in [0, 1].
    """

    rho: float
    eta: float
    gamma: float

    def validate(self) -> None:
        if not (-0.999 < self.rho < 0.999):
            raise ValueError("rho must lie inside (-0.999, 0.999).")
        if self.eta <= 0:
            raise ValueError("eta must be strictly positive.")
        if not (0 <= self.gamma <= 1):
            raise ValueError("gamma must lie in [0, 1].")


@dataclass(frozen=True)
class SliceDiagnostics:
    """
    Diagnostics computed for each observed slice under calibrated parameters.
    """

    expiry: pd.Timestamp
    tau: float
    theta: float
    psi_obs: float
    psi_model: float
    p_obs: float
    p_model: float
    c_obs: float
    c_model: float
    abs_error_psi: float
    abs_error_p: float
    abs_error_c: float
    rel_error_psi: float
    rel_error_p: float
    rel_error_c: float
    arbitrage_margin: float
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationResult:
    """
    Encapsulates the outcome of a calibration run.
    """

    params: SSVIParameters
    diagnostics: List[SliceDiagnostics]
    objective_value: float
    status: str
    message: str

    @property
    def max_abs_errors(self) -> Dict[str, float]:
        return {
            "psi": max(d.abs_error_psi for d in self.diagnostics),
            "p": max(d.abs_error_p for d in self.diagnostics),
            "c": max(d.abs_error_c for d in self.diagnostics),
        }

    @property
    def max_rel_errors(self) -> Dict[str, float]:
        return {
            "psi": max(d.rel_error_psi for d in self.diagnostics),
            "p": max(d.rel_error_p for d in self.diagnostics),
            "c": max(d.rel_error_c for d in self.diagnostics),
        }


# ---------------------------------------------------------------------------
# Protocols (interfaces for dependency inversion)
# ---------------------------------------------------------------------------


class  SliceProvider(Protocol):
    """
    Supplies observed slices to the calibrator.
    """

    def get_slices(self) -> Sequence[ObservedSSVISlice]:
        ...


class LossFunction(Protocol):
    """
    Computes scalar loss for given parameters and slices.
    """

    def __call__(self, params_vec: np.ndarray, slices: Sequence[ObservedSSVISlice]) -> float:
        ...


class OptimizationBackend(Protocol):
    """
    Strategy interface for numerical optimization.
    """

    def minimize(
        self,
        fun: callable,
        x0: np.ndarray,
        bounds: Sequence[Tuple[float, float]],
        **kwargs,
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        ...


# ---------------------------------------------------------------------------
# Core SSVI computations
# ---------------------------------------------------------------------------


class SSVIModel:
    """
    SSVI surface evaluator with immutable parameters.
    """

    def __init__(self, params: SSVIParameters):
        params.validate()
        self._params = params

    @property
    def params(self) -> SSVIParameters:
        return self._params

    def phi(self, theta: float) -> float:
        """
        Shape function φ(θ) = η θ^{-γ}
        """
        if theta <= 0:
            raise ValueError("theta must be positive.")
        eta, gamma = self._params.eta, self._params.gamma
        return eta * theta ** (-gamma)

    def psi(self, theta: float) -> float:
        """
        ψ(θ) = θ φ(θ)
        """
        return theta * self.phi(theta)

    def wing_slopes(self, theta: float) -> Tuple[float, float]:
        """
        Compute left/right wing slopes (p, c) implied by the SSVI parameters.
        """
        psi_val = self.psi(theta)
        rho = self._params.rho
        common = 0.5 * psi_val
        return common * (1 - rho), common * (1 + rho)

    def total_variance(self, log_moneyness: float, theta: float) -> float:
        """
        SSVI total variance w(k, θ).
        """
        rho = self._params.rho
        phi_theta = self.phi(theta)
        x = phi_theta * log_moneyness + rho
        root = math.sqrt(x * x + (1 - rho * rho))
        return 0.5 * theta * (1 + rho * phi_theta * log_moneyness + root)

    def implied_vol(
        self,
        forward: float,
        strike: float,
        tau: float,
        theta: float,
    ) -> float:
        """
        Convert total variance to Black implied volatility.
        """
        if tau <= 0:
            raise ValueError("tau must be positive.")
        if forward <= 0 or strike <= 0:
            raise ValueError("forward and strike must be positive.")

        k = math.log(strike / forward)
        total_var = self.total_variance(k, theta)
        return math.sqrt(max(total_var / tau, 0.0))


# ---------------------------------------------------------------------------
# Arbitrage diagnostics
# ---------------------------------------------------------------------------


class ArbitrageConditions:
    """
    Analytical constraints ensuring absence of static arbitrage in SSVI.
    """

    @staticmethod
    def butterfly_margin(theta: float, psi: float, rho: float) -> float:
        """
        SSVI butterfly condition requires:
            psi <= 4 / (1 + |rho|)
        Returns positive margin if condition satisfied, negative otherwise.
        """
        if theta <= 0:
            return float("-inf")
        limit = 4.0 / (1.0 + abs(rho))
        return limit - psi

    @staticmethod
    def calendar_margin(
        slice_a: ObservedSSVISlice,
        slice_b: ObservedSSVISlice,
    ) -> float:
        """
        Calendar spread condition between earlier slice_a (tau_a < tau_b)
        and later slice_b. Uses Gatheral-Jacquier inequality:
            (psi_b - psi_a) / (theta_b - theta_a) * (1 + |rho|)
            <= 1 / min(theta_a, theta_b)
        Returns minimal margin across both differences.
        """
        if slice_b.tau <= slice_a.tau:
            raise ValueError("Slices must be ordered by increasing expiry.")

        delta_theta = slice_b.theta - slice_a.theta
        delta_psi = slice_b.psi - slice_a.psi

        if delta_theta <= 0 or delta_psi < 0:
            return float("-inf")

        min_theta = min(slice_a.theta, slice_b.theta)
        if min_theta <= 0:
            return float("-inf")

        rho_hat = slice_b.observed_rho
        if np.isnan(rho_hat):
            rho_hat = slice_a.observed_rho

        if np.isnan(rho_hat):
            rho_hat = 0.0

        lhs = delta_psi / delta_theta * (1 + abs(rho_hat))
        rhs = 4.0 / min_theta
        return rhs - lhs

    @staticmethod
    def calendar_margin_2(
        slice_a: ObservedSSVISlice,
        slice_b: ObservedSSVISlice,
    ) -> float:
        """
        Alternative calendar spread condition between earlier slice_a (tau_a < tau_b)
        and later slice_b.
        Returns minimal margin across both differences.
        """
        if slice_b.tau <= slice_a.tau:
            raise ValueError("Slices must be ordered by increasing expiry.")

        if slice_a.theta <= 0 or slice_b.theta <= 0:
            return float("-inf")

        return slice_a.theta - slice_b.theta


# ---------------------------------------------------------------------------
# Loss Function Implementations
# ---------------------------------------------------------------------------


@dataclass
class LossWeights:
    """
    Per-term weights for the calibration loss.
    """

    psi: float = 1.0
    p: float = 1.0
    c: float = 1.0


class WeightedRelativeSquaredErrorLoss:
    """
    Weighted sum of squared relative errors for (psi, p, c).
    """

    def __init__(self, weights: LossWeights):
        self._weights = weights

    def __call__(self, params_vec: np.ndarray, slices: Sequence[ObservedSSVISlice]) -> float:
        params = SSVIParameters(rho=float(params_vec[0]), eta=float(params_vec[1]), gamma=float(params_vec[2]))
        try:
            params.validate()
        except ValueError:
            return float("inf")

        model = SSVIModel(params)
        loss = 0.0

        for slc in slices:
            psi_model = model.psi(slc.theta)
            p_model, c_model = model.wing_slopes(slc.theta)

            loss += self._weights.psi * self._relative_squared_error(psi_model, slc.psi)
            loss += self._weights.p * self._relative_squared_error(p_model, slc.p)
            loss += self._weights.c * self._relative_squared_error(c_model, slc.c)

            butterfly_margin = ArbitrageConditions.butterfly_margin(slc.theta, psi_model, params.rho)
            if butterfly_margin < 0:
                # Penalize butterfly arbitrage violations heavily
                loss += 1e6 * abs(butterfly_margin)
            

        return float(loss)

    @staticmethod
    def _relative_squared_error(model_val: float, obs_val: float) -> float:
        if obs_val == 0:
            return (model_val - obs_val) ** 2
        return ((model_val - obs_val) / obs_val) ** 2


# ---------------------------------------------------------------------------
# Optimization backends
# ---------------------------------------------------------------------------


class ScipyOptimizationBackend:
    """
    SciPy-based optimizer implementing the OptimizationBackend protocol.
    """

    def __init__(self, method: str = "Nelder-Mead", maxiter: int = 1000, tol: float = 1e-9):
        if minimize is None:
            raise ImportError("scipy is required for ScipyOptimizationBackend.")
        self._method = method
        self._maxiter = maxiter
        self._tol = tol

    def minimize(
        self,
        fun: callable,
        x0: np.ndarray,
        bounds: Sequence[Tuple[float, float]],
        **kwargs,
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        result = minimize(
            fun,
            x0=x0,
            method=self._method,
            bounds=bounds,
            tol=self._tol,
            options={"maxiter": self._maxiter},
        )
        info = {"converged": bool(result.success), "nit": result.nit, "status": result.status}
        return result.x, float(result.fun), info


# class GridSearchOptimizationBackend:
#     """
#     Deterministic fallback optimizer for environments without SciPy.
#     Performs a coarse grid search followed by a local refinement (optional).
#     """

#     def __init__(
#         self,
#         rho_grid: Iterable[float] = np.linspace(-0.8, 0.8, 21),
#         eta_grid: Iterable[float] = np.linspace(0.01, 10.0, 30),
#         gamma_grid: Iterable[float] = np.linspace(0.0, 1.0, 21),
#     ):
#         self._rho_grid = list(rho_grid)
#         self._eta_grid = list(eta_grid)
#         self._gamma_grid = list(gamma_grid)

#     def minimize(
#         self,
#         fun: callable,
#         x0: np.ndarray,
#         bounds: Sequence[Tuple[float, float]],
#         **kwargs,
#     ) -> Tuple[np.ndarray, float, Dict[str, float]]:
#         best_x = None
#         best_val = float("inf")
#         evaluations = 0

#         for rho in self._rho_grid:
#             for eta in self._eta_grid:
#                 for gamma in self._gamma_grid:
#                     x = np.array([rho, eta, gamma], dtype=float)
#                     if not self._within_bounds(x, bounds):
#                         continue
#                     val = fun(x)
#                     evaluations += 1
#                     if val < best_val:
#                         best_val = val
#                         best_x = x

#         if best_x is None:
#             raise CalibrationError("Grid search failed to evaluate any feasible points.")

#         info = {"converged": True, "evaluations": evaluations, "status": 0}
#         return best_x, best_val, info

    @staticmethod
    def _within_bounds(x: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> bool:
        for val, (lower, upper) in zip(x, bounds):
            if (lower is not None and val < lower) or (upper is not None and val > upper):
                return False
        return True


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class SSVICalibrator:
    """
    Orchestrates calibration of global SSVI parameters.
    """

    def __init__(
        self,
        slice_provider: SliceProvider,
        loss_function: LossFunction,
        optimizer: OptimizationBackend,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        self._slice_provider = slice_provider
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._bounds = bounds or [(-0.999, 0.999), (1e-4, 50.0), (0.0, 1.0)]

    def calibrate(
        self,
        initial_guess: Optional[Tuple[float, float, float]] = None,
    ) -> CalibrationResult:
        """
        Run numerical calibration and assemble diagnostic outputs.
        """
        slices = list(self._validated_slices())

        if not slices:
            raise CalibrationError("No slices available for calibration.")

        x0 = np.array(initial_guess if initial_guess is not None else self._default_initial_guess(slices))
        solution, objective_value, info = self._optimizer.minimize(
            fun=lambda x: self._loss_function(x, slices),
            x0=x0,
            bounds=self._bounds,
        )
        # solution, objective_value, info = minimize(
        #     fun=lambda x: self._loss_function(x, slices),
        #     x0=x0,
        #     bounds=self._bounds,
        # )


        params = SSVIParameters(rho=float(solution[0]), eta=float(solution[1]), gamma=float(solution[2]))
        params.validate()

        diagnostics = self._compute_diagnostics(params, slices)
        status = "success" if info.get("converged", False) else "warning"

        return CalibrationResult(
            params=params,
            diagnostics=diagnostics,
            objective_value=objective_value,
            status=status,
            message=f"Optimizer status {info.get('status', 'n/a')} after {info.get('nit', info.get('evaluations', 'n/a'))} iterations.",
        )

    def _validated_slices(self) -> Iterable[ObservedSSVISlice]:
        for slc in self._slice_provider.get_slices():
            slc.validate()
            yield slc

    @staticmethod
    def _default_initial_guess(slices: Sequence[ObservedSSVISlice]) -> Tuple[float, float, float]:
        rho_guess = np.clip(np.nanmedian([s.observed_rho for s in slices if not np.isnan(s.observed_rho)]), -0.8, 0.8)
        theta_vals = np.array([s.theta for s in slices])
        psi_vals = np.array([s.psi for s in slices])
        eta_guess = max(np.median(psi_vals / theta_vals), 0.1)
        gamma_guess = 0.5
        return float(rho_guess), float(eta_guess), float(gamma_guess)

    def _compute_diagnostics(
        self, params: SSVIParameters, slices: Sequence[ObservedSSVISlice]
    ) -> List[SliceDiagnostics]:
        model = SSVIModel(params)
        diagnostics: List[SliceDiagnostics] = []

        for slc in slices:
            psi_model = model.psi(slc.theta)
            p_model, c_model = model.wing_slopes(slc.theta)

            diagnostics.append(
                SliceDiagnostics(
                    expiry=slc.expiry,
                    tau=slc.tau,
                    theta=slc.theta,
                    psi_obs=slc.psi,
                    psi_model=psi_model,
                    p_obs=slc.p,
                    p_model=p_model,
                    c_obs=slc.c,
                    c_model=c_model,
                    abs_error_psi=abs(psi_model - slc.psi),
                    abs_error_p=abs(p_model - slc.p),
                    abs_error_c=abs(c_model - slc.c),
                    rel_error_psi=self._safe_relative_error(psi_model, slc.psi),
                    rel_error_p=self._safe_relative_error(p_model, slc.p),
                    rel_error_c=self._safe_relative_error(c_model, slc.c),
                    arbitrage_margin=ArbitrageConditions.butterfly_margin(slc.theta, psi_model, params.rho),
                    metadata=slc.metadata,
                )
            )
        return diagnostics

    @staticmethod
    def _safe_relative_error(model_val: float, obs_val: float) -> float:
        if obs_val == 0:
            return float("inf") if model_val != 0 else 0.0
        return abs((model_val - obs_val) / obs_val)


# ---------------------------------------------------------------------------
# Data Access Helpers
# ---------------------------------------------------------------------------


class DataFrameSliceProvider:
    """
    Concrete SliceProvider pulling slices from a pandas DataFrame. The DataFrame
    must contain the columns listed in `required_columns`. Column names can be
    remapped via `column_map`.
    """

    required_columns: Tuple[str, ...] = (
        "Expiry",
        "TTE_years",
        "theta",
        "psi",
        "p_t",
        "c_t",
        "svi_rho",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        column_map: Optional[Dict[str, str]] = None,
        metadata_columns: Optional[Sequence[str]] = None,
        preprocess: bool = True,
    ):
        self._data = data.copy()
        self._column_map = column_map or {}
        self._metadata_columns = metadata_columns or []
        if preprocess:
            self._preprocess()

    def get_slices(self) -> Sequence[ObservedSSVISlice]:
        slices: List[ObservedSSVISlice] = []
        for _, row in self._data.iterrows():
            metadata = {col: row[col] for col in self._metadata_columns if col in row}
            slice_obj = ObservedSSVISlice(
                expiry=pd.to_datetime(row["Expiry"]),
                time_to_expiry=float(row["TTE_years"]),
                theta=float(row["theta"]),
                psi=float(row["psi"]),
                p=float(row["p_t"]),
                c=float(row["c_t"]),
                svi_rho=float(row["svi_rho"]),
                metadata=metadata,
            )
            slices.append(slice_obj)
        return slices

    # ---- internal helpers -------------------------------------------------

    def _preprocess(self) -> None:
        self._rename_columns()
        self._ensure_required_columns()
        self._data = self._data.dropna(subset=self.required_columns, how="any")
        self._data = self._data.sort_values(by="TTE_years").reset_index(drop=True)

    def _rename_columns(self) -> None:
        if not self._column_map:
            return
        self._data = self._data.rename(columns=self._column_map)

    def _ensure_required_columns(self) -> None:
        missing = [col for col in self.required_columns if col not in self._data.columns]
        if missing:
            raise KeyError(f"Missing required columns for SSVI slices: {missing}")


# ---------------------------------------------------------------------------
# Factory functions for orchestrators
# ---------------------------------------------------------------------------


def build_default_calibrator(
    df: pd.DataFrame,
    *,
    column_map: Optional[Dict[str, str]] = None,
    metadata_columns: Optional[Sequence[str]] = None,
    optimizer: Optional[OptimizationBackend] = None,
    loss_weights: Optional[LossWeights] = None,
) -> Tuple[SSVICalibrator, SliceProvider]:
    """
    Convenience constructor assembling the standard calibrator.
    """
    slice_provider = DataFrameSliceProvider(df, column_map=column_map, metadata_columns=metadata_columns)
    loss_fn = WeightedRelativeSquaredErrorLoss(loss_weights or LossWeights())

    # if optimizer is None:
    #     if minimize is not None:
    #         optimizer = ScipyOptimizationBackend()
    #     else:
    #         warnings.warn("SciPy not available, falling back to grid-search optimizer.")
    #         optimizer = GridSearchOptimizationBackend()

    calibrator = SSVICalibrator(
        slice_provider=slice_provider,
        loss_function=loss_fn,
        optimizer=ScipyOptimizationBackend(),
    )
    return calibrator, slice_provider


def evaluate_surface_on_grid(
    model: SSVIModel,
    thetas: Sequence[float],
    log_moneyness_grid: Sequence[float],
) -> pd.DataFrame:
    """
    Produce a tabular representation of total variances w(k, θ).
    """
    records = []
    for theta in thetas:
        for k in log_moneyness_grid:
            records.append(
                {
                    "theta": theta,
                    "log_moneyness": k,
                    "total_variance": model.total_variance(k, theta),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_calibration(result: CalibrationResult) -> pd.DataFrame:
    """
    Convert diagnostics into a human-readable table.
    """
    return pd.DataFrame(
        [
            {
                "Expiry": d.expiry,
                "Tau": d.tau,
                "Theta": d.theta,
                "Psi_obs": d.psi_obs,
                "Psi_model": d.psi_model,
                "Psi_rel_err": d.rel_error_psi,
                "P_obs": d.p_obs,
                "P_model": d.p_model,
                "P_rel_err": d.rel_error_p,
                "C_obs": d.c_obs,
                "C_model": d.c_model,
                "C_rel_err": d.rel_error_c,
                "Arb_margin": d.arbitrage_margin,
                **d.metadata,
            }
            for d in result.diagnostics
        ]
    )


__all__ = [
    # Exceptions
    "CalibrationError",
    "InvalidSliceError",
    # Domain models
    "ObservedSSVISlice",
    "SSVIParameters",
    "SliceDiagnostics",
    "CalibrationResult",
    # Helpers and services
    "LossWeights",
    "WeightedRelativeSquaredErrorLoss",
    "ScipyOptimizationBackend",
    "GridSearchOptimizationBackend",
    "SSVIModel",
    "ArbitrageConditions",
    "SSVICalibrator",
    "DataFrameSliceProvider",
    "build_default_calibrator",
    "evaluate_surface_on_grid",
    "summarize_calibration",
]