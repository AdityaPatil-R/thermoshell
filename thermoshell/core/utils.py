import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List, Dict

# --- 1. Matrix Math Utilities ---

def mmt(matrix: np.ndarray) -> np.ndarray:
    """
    Computes the matrix plus its transpose: (M + M^T).
    Used to ensure the resulting stiffness (Hessian) matrix is symmetric
    during element assembly.
    """
    return matrix + matrix.T

# --- 2. CLI Argument Parsing ---

def parse_args():
    """
    Parses command-line arguments and returns an object containing
    CLI overrides for simulation parameters. These flags will be used 
    in main.py to override default values in the dataclasses.
    """
    p = argparse.ArgumentParser(description="ThermoShell FE Simulation")
    
    # Core Overrides
    p.add_argument('--mesh', type=int, help="Which mesh ID to load (GeomParams.mesh_id).")
    p.add_argument('--eps-thermal', type=float, help="Thermal strain magnitude (MaterialParams.thermal_strain_mag).")
    p.add_argument('--gravity', type=int, choices=[0, 1], help="0: Disable gravity, 1: Enable gravity (SimParams.use_gravity).")
    p.add_argument('--fluctuate', type=int, choices=[0, 1], help="0: No boundary fluctuation, 1: Add boundary fluctuation (SimParams.use_fluctuations).")
    
    # Console Flags (Set in SimParams/overridden here)
    p.add_argument('--print', dest='do_print', action='store_true', help="Enable verbose console output (SimParams.do_print).")
    p.add_argument('--plot', dest='do_plot', action='store_true', help="Enable final mesh plots (SimParams.do_plot).")
    p.add_argument('--test', dest='do_test', action='store_true', help="Enable analytical gradient/Hessian verification tests (SimParams.do_test).")
    
    return p.parse_args()

# --- 3. Plotting Utilities ---

def new_fig(num: int = None, figsize: Tuple[float, float] = (6, 6),
            label_fmt: str = 'Fig. {n}', label_pos: Tuple[float, float] = (0.01, 1.00), **subplot_kw) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create (or switch to) figure `num` and a single Axes,
    then label the figure in the upper-left corner.
    
    Parameters:
        num (int): Figure number (or None for auto-increment).
        figsize (Tuple): Passed to plt.figure.
        label_fmt (str): Format string for the figure label.
        label_pos (Tuple): (x, y) in figure fraction (0-1) for the label.
        **subplot_kw: Passed to fig.add_subplot (e.g., projection='3d').
        
    Returns:
        (fig, ax): The new or current Figure and Axes object.
    """
    fig = plt.figure(num, figsize=figsize)
    # Clear existing content if re-using a figure number
    fig.clf()
    
    ax = fig.add_subplot(1, 1, 1, **subplot_kw)
    
    # Place label in figure coordinates
    lbl = label_fmt.format(n=fig.number)
    fig.text(
        label_pos[0], label_pos[1], lbl,
        transform=fig.transFigure,
        fontsize=12, fontweight='bold',
        va='top'
    )
    return fig, ax
