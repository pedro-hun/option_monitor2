# ssvi/visualization.py
from typing import Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_ssvi_slices(model, expiries, k_min=-0.8, k_max=0.8,
                     num_points=200,
                     market_points=()):
    """Matplotlib slice plot (see earlier snippet)."""
    k_grid = np.linspace(k_min, k_max, num_points)
    plt.figure(figsize=(10, 6))
    for t in expiries:
        vols = np.array([model.implied_vol(k, t) for k in k_grid])
        plt.plot(k_grid, vols, label=f"SSVI t={t:.3f}y")
    for t, k_obs, sigma_obs in market_points:
        plt.scatter(k_obs, sigma_obs, s=15, alpha=0.6,
                    label=f"Market t={t:.3f}y")
    plt.xlabel("Log-moneyness k = ln(K/F)")
    plt.ylabel("Implied volatility")
    plt.title("SSVI vs Market slices")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)
    plt.show()

def plot_ssvi_heatmap(model, t_min, t_max, k_min, k_max,
                      num_t=40, num_k=120):
    """Heat map of implied vols."""
    times = np.linspace(t_min, t_max, num_t)
    ks = np.linspace(k_min, k_max, num_k)
    sigma_grid = np.empty((num_t, num_k))
    for i, t in enumerate(times):
        sigma_grid[i, :] = [model.implied_vol(k, t) for k in ks]
    plt.figure(figsize=(11, 6))
    plt.pcolormesh(ks, times, sigma_grid, shading="auto", cmap="viridis")
    plt.colorbar(label="Implied vol")
    plt.xlabel("Log-moneyness k")
    plt.ylabel("Time to expiry (years)")
    plt.title("SSVI implied volatility surface")
    plt.show()

def plot_ssvi_surface_3d(model, t_min, t_max, k_min, k_max,
                         num_t=40, num_k=120):
    """Interactive Plotly surface."""
    times = np.linspace(t_min, t_max, num_t)
    ks = np.linspace(k_min, k_max, num_k)
    sigma_grid = [
        [model.implied_vol(k, t) for k in ks]
        for t in times
    ]
    fig = go.Figure(data=go.Surface(
        x=ks, y=times, z=sigma_grid,
        colorscale="Viridis", showscale=True))
    fig.update_layout(
        title="SSVI implied volatility surface",
        scene=dict(
            xaxis_title="Log-moneyness k",
            yaxis_title="Time to expiry (years)",
            zaxis_title="Implied vol"
        )
    )
    fig.show()