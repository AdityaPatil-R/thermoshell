# thermoshell2/config.py

import numpy as np
from typing import Dict, Tuple

# --- Command Line Interface (CLI) Setup ---
# Default initial flags (will be updated by CLI/main_runner logic)
iPrint = 0
iPlot = 0
iTest = 0
eps_thermal = -0.3
iMesh = 1
iGravity = 1
iFluc = 0

# --- Mesh Definitions ---
mesh_files: Dict[int, str] = {
    1: '../../data/mesh_python_circle_970nodes_scale100mm.txt',
    2: '../../data/mesh_rectangle_scaled_1215nodes_scale100mm_155mm.txt',
    3: '../../data/mesh_python_square_388nodes_scale100mm.txt',
}

# --- Default Geometry/Pattern Constants (MUST be set here for initial defaults) ---
Yratio: float = 1.0        # Ratio between in and out Y values.
OuterR: float = 0.05       # Outer radius of circular plate.
# iMesh = 1 / iMesh = 3 specific constants
StripeWidth: float = 0.0059
StripeLength: float = 0.0441  # Will be overwritten to 0.0707 for iMesh=3
Stripe_r: float = 0.0029      # Will be overwritten to 0.0006 for iMesh=3
# iMesh = 2 specific constants
delta_shape: float = 0.02787
n_spokes: int = 6
star_radius_val: float = 0.044595 # Single float value
star_thickness_val: float = 0.002787
beam_thickness_val: float = 0.002787

# Magnitude percentage of fluctuation
epsilon_th_fluctuation: float = 2.0
eps_thermal_min: float = 0

# --- Material Constants ---
Ysoft: float = 1.0e6
Yhard: float = 3.0e6
FactorKs: float = 10.0
FactorKb: float = 1.0

# Layer thicknesses: (lk, h1, h2)
PARAMS: Dict[int, Tuple[float, float, float]] = {
    1: (0.0032, 0.3e-3, 0.7e-3),
    2: (0.0040, 0.3e-3, 0.6e-3),
    3: (0.0058, 0.3e-3, 1.0e-3),
}

# Calculated parameters (placeholders, updated in main_runner)
lk, h1, h2 = PARAMS[iMesh]

# --- Simulation Parameters ---
totalTime: float = 1.0
dt_min: float = 1e-8
dt_max: float = 0.1
dt: float = 0.05
qtol: float = 1e-5
maxIter: int = 20
beta: float = 1.0
theta_bar: float = 0.0
model_choice: int = 1
n_record: int = 2

# Initial Gravity vector
g_vec_default: np.ndarray = np.array([0.0, 0.0, -9.81])