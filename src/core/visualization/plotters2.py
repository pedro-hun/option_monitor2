import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Optional
from mpl_toolkits.mplot3d import Axes3D

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.core.surfaces.ssvi2 import SSVIModel, CalibrationResult, ObservedSSVISlice


def plot_ssvi_slices_2d(
    model: SSVIModel,
    theta_values: Sequence[float],
    k_min: float = -1.0,
    k_max: float = 1.0,
    num_points: int = 200,
    market_slices: Optional[Sequence[ObservedSSVISlice]] = None,
    title: str = "SSVI Volatility Slices"
) -> None:
    """Plot SSVI implied volatility slices as 2D lines."""
    
    k_grid = np.linspace(k_min, k_max, num_points)
    
    plt.figure(figsize=(12, 8))
    
    # Plot model curves
    for theta in theta_values:
        vols = []
        for k in k_grid:
            try:
                total_var = model.total_variance(k, theta)
                # Convert total variance to implied volatility
                # Using theta as proxy for tau (time to expiry)
                tau = theta / 0.04  # Assuming ATM vol around 20% (0.04 = 0.2^2)
                vol = np.sqrt(max(total_var / tau, 0))
                vols.append(vol)
            except:
                vols.append(np.nan)
        
        plt.plot(k_grid, vols, label=f'θ = {theta:.4f}', linewidth=2)
    
    # Plot market data if available
    if market_slices:
        theta_to_slices = {}
        for slice_obj in market_slices:
            theta = slice_obj.theta
            if theta not in theta_to_slices:
                theta_to_slices[theta] = []
            theta_to_slices[theta].append(slice_obj)
        
        for theta, slices in theta_to_slices.items():
            # For demonstration, create some sample strikes around ATM
            strikes = np.linspace(-0.5, 0.5, 5)
            market_vols = []
            for k in strikes:
                # Approximate volatility from SSVI slice parameters
                # This is a simplified approach for visualization
                tau = theta / 0.04
                vol = np.sqrt(theta / tau) * (1 + 0.1 * k)  # Simple approximation
                market_vols.append(max(vol, 0.05))
            
            plt.scatter(strikes, market_vols, 
                       alpha=0.7, s=50, 
                       label=f'Market θ = {theta:.4f}')
    
    plt.xlabel('Log-moneyness (k)', fontsize=12)
    plt.ylabel('Implied Volatility', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ssvi_surface_3d_matplotlib(
    model: SSVIModel,
    theta_range: Tuple[float, float] = (0.01, 0.5),
    k_range: Tuple[float, float] = (-1.0, 1.0),
    num_theta: int = 30,
    num_k: int = 50
) -> None:
    """Plot 3D volatility surface using matplotlib."""
    
    theta_grid = np.linspace(theta_range[0], theta_range[1], num_theta)
    k_grid = np.linspace(k_range[0], k_range[1], num_k)
    
    K_mesh, Theta_mesh = np.meshgrid(k_grid, theta_grid)
    Vol_surface = np.zeros_like(K_mesh)
    
    for i, theta in enumerate(theta_grid):
        for j, k in enumerate(k_grid):
            try:
                total_var = model.total_variance(k, theta)
                tau = theta / 0.04  # Convert theta to time
                vol = np.sqrt(max(total_var / tau, 0))
                Vol_surface[i, j] = vol
            except:
                Vol_surface[i, j] = np.nan
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(K_mesh, Theta_mesh, Vol_surface, 
                          cmap='viridis', alpha=0.9,
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('Log-moneyness (k)', fontsize=12)
    ax.set_ylabel('Theta (θ)', fontsize=12)
    ax.set_zlabel('Implied Volatility', fontsize=12)
    ax.set_title('SSVI Volatility Surface', fontsize=14)
    
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()


def plot_ssvi_surface_3d_plotly(
    model: SSVIModel,
    theta_range: Tuple[float, float] = (0.01, 0.5),
    k_range: Tuple[float, float] = (-1.0, 1.0),
    num_theta: int = 40,
    num_k: int = 60
) -> None:
    """Plot interactive 3D volatility surface using plotly."""
    
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Using matplotlib instead.")
        plot_ssvi_surface_3d_matplotlib(model, theta_range, k_range, num_theta, num_k)
        return
    
    theta_grid = np.linspace(theta_range[0], theta_range[1], num_theta)
    k_grid = np.linspace(k_range[0], k_range[1], num_k)
    
    Vol_surface = []
    for theta in theta_grid:
        vol_row = []
        for k in k_grid:
            try:
                total_var = model.total_variance(k, theta)
                tau = theta / 0.04
                vol = np.sqrt(max(total_var / tau, 0))
                vol_row.append(vol)
            except:
                vol_row.append(np.nan)
        Vol_surface.append(vol_row)
    
    fig = go.Figure(data=[go.Surface(
        x=k_grid,
        y=theta_grid,
        z=Vol_surface,
        colorscale='Viridis',
        showscale=True
    )])
    
    fig.update_layout(
        title='SSVI Volatility Surface',
        scene=dict(
            xaxis_title='Log-moneyness (k)',
            yaxis_title='Theta (θ)',
            zaxis_title='Implied Volatility'
        ),
        width=800,
        height=600
    )
    
    fig.show()


def plot_calibration_diagnostics(result: CalibrationResult) -> None:
    """Plot calibration fit quality diagnostics."""
    
    diagnostics = result.diagnostics
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    expiries = [d.tau for d in diagnostics]
    psi_obs = [d.psi_obs for d in diagnostics]
    psi_model = [d.psi_model for d in diagnostics]
    p_obs = [d.p_obs for d in diagnostics]
    p_model = [d.p_model for d in diagnostics]
    c_obs = [d.c_obs for d in diagnostics]
    c_model = [d.c_model for d in diagnostics]
    
    # Psi comparison
    axes[0, 0].scatter(psi_obs, psi_model, alpha=0.7, s=50)
    axes[0, 0].plot([min(psi_obs), max(psi_obs)], [min(psi_obs), max(psi_obs)], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Observed Psi')
    axes[0, 0].set_ylabel('Model Psi')
    axes[0, 0].set_title('Psi: Model vs Observed')
    axes[0, 0].grid(True, alpha=0.3)
    
    # P comparison
    axes[0, 1].scatter(p_obs, p_model, alpha=0.7, s=50, color='orange')
    axes[0, 1].plot([min(p_obs), max(p_obs)], [min(p_obs), max(p_obs)], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('Observed P')
    axes[0, 1].set_ylabel('Model P')
    axes[0, 1].set_title('P: Model vs Observed')
    axes[0, 1].grid(True, alpha=0.3)
    
    # C comparison
    axes[1, 0].scatter(c_obs, c_model, alpha=0.7, s=50, color='green')
    axes[1, 0].plot([min(c_obs), max(c_obs)], [min(c_obs), max(c_obs)], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Observed C')
    axes[1, 0].set_ylabel('Model C')
    axes[1, 0].set_title('C: Model vs Observed')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Relative errors by expiry
    psi_rel_errors = [d.rel_error_psi for d in diagnostics]
    p_rel_errors = [d.rel_error_p for d in diagnostics]
    c_rel_errors = [d.rel_error_c for d in diagnostics]
    
    axes[1, 1].plot(expiries, psi_rel_errors, 'o-', label='Psi', alpha=0.8)
    axes[1, 1].plot(expiries, p_rel_errors, 's-', label='P', alpha=0.8)
    axes[1, 1].plot(expiries, c_rel_errors, '^-', label='C', alpha=0.8)
    axes[1, 1].set_xlabel('Time to Expiry')
    axes[1, 1].set_ylabel('Relative Error')
    axes[1, 1].set_title('Relative Errors by Expiry')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== CALIBRATION SUMMARY ===")
    print(f"Status: {result.status}")
    print(f"Objective Value: {result.objective_value:.6f}")
    print(f"Parameters: rho={result.params.rho:.4f}, eta={result.params.eta:.4f}, gamma={result.params.gamma:.4f}")
    print(f"\nMax Absolute Errors: {result.max_abs_errors}")
    print(f"Max Relative Errors: {result.max_rel_errors}")