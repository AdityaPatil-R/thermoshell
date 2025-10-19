import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import argparse
from typing import Dict, Tuple
from functools import partial
from itertools import combinations
from collections import defaultdict, Counter

# --- Import modules from the new package structure ---
import config as cfg

# Geometry and Plotting (Already Correctly Updated)
from geometry.mesh_io import load_mesh
from geometry.mesh_props import fun_edge_lengths
from viz.figure_setup import new_fig 
from viz.mesh_plots import plot_truss_3d
from viz.thermal_plots import (
    plot_thermal_strain_edges, 
    plot_thermal_strain_edges_CustomRange
)

from analysis.material.bilayer import (
    bilayer_flexural_rigidity
)

from analysis.material.assignment import (
    assign_thermal_strains_contour
)

from analysis.material.assignment import (
    assign_youngs_modulus,
    assign_youngs_modulus_v3
)

from analysis.patterning.regions import (
    circle_six_arms_region, 
    square_X_region
)

from analysis.patterning.complex import (
    whole_peanut_region
)

# Finite Element Assembly (MODIFIED)
from assembly.assemblers import ElasticGHEdgesCoupledThermal

# Solver components (Unchanged)
from solver.boundary_conditions import (
    BoundaryConditions3D, 
    fun_BC_3D_hold_center, 
    fun_BC_peanut
)
from solver.time_stepper import timeStepper3D_static, record_step 
from analysis.bending_model.geometry import getTheta


start = time.perf_counter()

# ==============================================================================
# 0. CLI, Configuration, and Initialization
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', type=int, help="which mesh to load")
    p.add_argument('--eps-thermal', type=float, help="thermal strain")
    p.add_argument('--print', dest='do_print', action='store_true')
    p.add_argument('--plot', dest='do_plot', action='store_true')
    p.add_argument('--test', dest='do_test', action='store_true')
    p.add_argument('--gravity', type=int, help="0=no gravity, 1=gravity")
    p.add_argument('--fluctuate', type=int, help="0=no, 1=yes")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Apply CLI overrides to config module variables
    if args.mesh is not None: cfg.iMesh = args.mesh
    if args.eps_thermal is not None: cfg.eps_thermal = args.eps_thermal
    if args.do_print: cfg.iPrint = 1
    if args.do_plot: cfg.iPlot = 1
    if args.do_test: cfg.iTest = 1
    if args.gravity is not None: cfg.iGravity = args.gravity
    if args.fluctuate is not None: cfg.iFluc = args.fluctuate

    print("CLI overrides:", {k: v for k, v in vars(args).items() if v is not None and v is not False})
    
    # --- CRITICAL: Replicate original variable re-assignments based on iMesh ---
    # These assignments must happen AFTER CLI parsing but BEFORE stiffness calc.
    if cfg.iMesh == 1:
        cfg.Yratio = 1.0
        cfg.OuterR = 0.05
        cfg.StripeWidth=0.0059
        cfg.StripeLength=0.0441
        cfg.Stripe_r=0.0029
    elif cfg.iMesh == 2:
        cfg.Yratio = 1.0
        cfg.OuterR = 0.05
        # delta_shape, n_spokes, etc. are already set in config.py
    elif cfg.iMesh == 3:
        cfg.Yratio = 1.0
        cfg.OuterR = 0.05
        cfg.StripeWidth=0.0059
        cfg.StripeLength=0.0707
        cfg.Stripe_r=0.0006
    
    # Update core stiffness parameters using final iMesh from PARAMS dictionary
    try:
        cfg.lk, cfg.h1, cfg.h2 = cfg.PARAMS[cfg.iMesh]
    except KeyError:
        raise ValueError(f"Unsupported mesh index: {cfg.iMesh!r}") from None
    
    eps_thermal_min = 0
    eps_thermal_max = cfg.eps_thermal


# ==============================================================================
# 1. Material Stiffness Calculations and Initial Output
# ==============================================================================

b1 = 1.0
D_1 = cfg.Ysoft * b1 * (cfg.h1**3) / 12.0
D_12 = bilayer_flexural_rigidity(cfg.h1, cfg.h2, cfg.Ysoft, cfg.Yhard)
kb1 = (2.0 / np.sqrt(3.0)) * D_1
kb12 = (2.0 / np.sqrt(3.0)) * D_12
ks1 = cfg.Ysoft * cfg.h1 * (cfg.lk**2) * np.sqrt(3) / 2.0
ks2 = cfg.Yhard * cfg.h2 * (cfg.lk**2) * np.sqrt(3) / 2.0
ks12 = ks1 + ks2

# Print the material constants (matches original output exactly)
print(f"Shrinky dink: D_1 = {D_1:.8f} per unit width")
print(f"Bilayer composite: D_12 = {D_12:.8f} per unit width")
print(f"kb1  = {kb1:.8f} N·m")
print(f"kb12  = {kb12:.8f} N·m")
print(f"kb12/kb1  = {kb12/kb1:.8f} N·m")
print(f"ks1 = {ks1:.8f} N·m")
print(f"ks2 = {ks2:.8f} N·m")
print(f"ks12 = {ks12:.8f} N·m")


# ==============================================================================
# 2. Mesh Loading and Pre-computation
# ==============================================================================

mesh_file = cfg.mesh_files[cfg.iMesh]
X0_4columns, ConnectivityMatrix_line, Triangles = load_mesh(mesh_file)

NP_total = X0_4columns.shape[0]
Nedges = ConnectivityMatrix_line.shape[0]
Ntriangles = Triangles.shape[0]
L0 = fun_edge_lengths(X0_4columns, ConnectivityMatrix_line)
X0 = X0_4columns[:, 1:4].ravel()
Ndofs = X0.size

node_xyz = X0_4columns[:, 1:4]
xs = node_xyz[:, 0]
ys = node_xyz[:, 1]
x_center = 0.5 * (xs.min() + xs.max())
y_center = 0.5 * (ys.min() + ys.max())
Pattern_center = (x_center, y_center)
OuterR = cfg.OuterR

# Print mesh info (matches original output)
print("nodeXYZ:\n", X0_4columns)
print("Connectivity:\n", ConnectivityMatrix_line)
print("Triangles:\n", Triangles)
print(f"Number of nodes:     {NP_total}")
print(f"Number of edges:     {Nedges}")
print(f"Number of triangles: {Ntriangles}")
print("Ref edge lengths:")
print("vector X0 =", X0)
print("Ndofs=", Ndofs)
print(f"Domain center: x={x_center:.4f}, y={y_center:.4f}")

# --- Hinge Quad Calculation (Original logic preserved) ---
tri_edges = []
for _, v1, v2, v3 in Triangles.astype(int):
    tri_edges += [tuple(sorted((v1, v2))), tuple(sorted((v2, v3))), tuple(sorted((v3, v1)))]
hinge_keys = {edge for edge, cnt in Counter(tri_edges).items() if cnt == 2}

edgeid_to_nodes = {int(eid): (int(n0), int(n1))
                   for eid, n0, n1 in ConnectivityMatrix_line.astype(int)}
edge_to_opps = defaultdict(list)
for _, v1, v2, v3 in Triangles.astype(int):
    verts = (v1, v2, v3)
    for a, b in combinations(verts, 2):
        key = tuple(sorted((a, b)))
        opp = next(v for v in verts if v not in key)
        edge_to_opps[key].append(opp)

hinge_edges = []
for eid, n0, n1 in ConnectivityMatrix_line.astype(int):
    if tuple(sorted((n0, n1))) in hinge_keys:
        hinge_edges.append(eid)
hinge_edges = np.array(hinge_edges, dtype=int)

hinge_quads = []
for eid in hinge_edges:
    n0, n1 = edgeid_to_nodes[eid]
    oppA, oppB = edge_to_opps[tuple(sorted((n0, n1)))]
    x0 = X0_4columns[n0, 1:4]; x1 = X0_4columns[n1, 1:4]
    x2 = X0_4columns[oppA, 1:4]; x3 = X0_4columns[oppB, 1:4]
    n0_v = np.cross(x1 - x0, x2 - x0)
    n1_v = np.cross(x3 - x0, x1 - x0)
    
    if (n0_v[2] < 0) and (n1_v[2] < 0):
        row = [eid, n0, n1, oppB, oppA]
    elif (n0_v[2] > 0) and (n1_v[2] > 0):
        row = [eid, n0, n1, oppA, oppB]
    else:
        row = [eid, n0, n1, oppA, oppB]
    hinge_quads.append(row)

HingeQuads_order = np.array(hinge_quads, dtype=int)
n_hinges = HingeQuads_order.shape[0]
hinge_eids = HingeQuads_order[:, 0]

# Print hinge info (matches original output)
print("Hinge edge IDs:", hinge_edges)
print("All edges:    ", np.arange(Nedges))
print("Hinge edges:  ", hinge_edges)
print("Boundary edges:", np.setdiff1d(np.arange(Nedges), hinge_edges))
print("edgeID, node0, node1, node2, node3")
print(HingeQuads_order[:3,:])
print("...")
print(HingeQuads_order[-3:,:])
print("edgeID, node0, node1, node2, node3 (ordered):")
print(HingeQuads_order[:3,:])
print("...")
print(HingeQuads_order[-3:,:])


# ==============================================================================
# 3. Pattern and Property Assignment
# ==============================================================================

# 3.1 Region function setup
if cfg.iMesh == 1:
    region_fn = partial(
        circle_six_arms_region,
        circle_center=Pattern_center,
        circle_radius=cfg.Stripe_r,
        arm_half_width=cfg.StripeWidth,
        arm_half_length=cfg.StripeLength
    )
elif cfg.iMesh == 2:
    # Handle the tuple/float confusion from the original script
    star_radius_f = cfg.star_radius_val
    star_thickness_f = cfg.star_thickness_val
    beam_thickness_f = cfg.beam_thickness_val
    
    region_fn = partial(
        whole_peanut_region,
        center=Pattern_center,
        delta_shape=cfg.delta_shape,
        star_radius=star_radius_f,
        star_thickness=star_thickness_f,
        n_spokes=cfg.n_spokes,
        beam_thickness=beam_thickness_f
    )
elif cfg.iMesh == 3:
    region_fn = partial(
        square_X_region,
        circle_center=Pattern_center,
        circle_radius=cfg.Stripe_r,
        arm_half_width=cfg.StripeWidth,
        arm_half_length=cfg.StripeLength
    )
else:
    raise ValueError(f"Unsupported mesh index: {cfg.iMesh}")

# 3.2 Assign initial thermal strains
eps_th_vector = assign_thermal_strains_contour(
    node_xyz,
    ConnectivityMatrix_line,
    cfg.eps_thermal,
    region_fn=region_fn,
    inside=False,
)
eps_th_vector_pattern = eps_th_vector.copy() # Store the pattern for scaling

# 3.3 Assign Young's Modulus array (Y_array)
if cfg.iMesh == 2:
    Y_array = assign_youngs_modulus_v3(
        node_xyz, ConnectivityMatrix_line, region_fn=region_fn,
        circle_center=Pattern_center, circle_radius=OuterR,
        Ysoft=cfg.Ysoft, Yhard=cfg.Yhard, Yratio=cfg.Yratio,
        inside=False, x_thresh_left=0.10, x_thresh_right=0.185, hard_factor=1.0,
    )
else:
    Y_array = assign_youngs_modulus(
        node_xyz, ConnectivityMatrix_line, region_fn=region_fn,
        circle_center=Pattern_center, circle_radius=OuterR,
        Ysoft=cfg.Ysoft, Yhard=cfg.Yhard, Yratio=cfg.Yratio,
        inside=False
    )
    
# 3.4 Calculate per-edge ks and per-hinge kb arrays
ks_array = np.where(Y_array == cfg.Yhard, ks12,
             np.where(Y_array == cfg.Ysoft, ks1, 0.0)) # 0.0 fallback from original code
kb_array = np.where(Y_array[hinge_eids] == cfg.Yhard, kb12,
                 np.where(Y_array[hinge_eids] == cfg.Ysoft, kb1, 0.0)) # 0.0 fallback

ks_array = ks_array * cfg.FactorKs
kb_array = kb_array * cfg.FactorKb


# --- Plotting Section (Must use iPlot variable from config) ---
if cfg.iPlot:
    
    # 2D Triangle Mesh plot (Original Figure 1, already executed, but redrawn here)
    # The original script shows this plot BEFORE all the material assignment prints.
    # To match, the core logic should replicate the original's flow:
    # 1. Print Material Constants. 2. Print Mesh Info. 3. Show Fig 1.
    
    plot_thermal_strain_edges(
        node_coords = node_xyz, connectivity=ConnectivityMatrix_line, 
        epsilon_th=eps_th_vector_pattern, title="thermal axial strain", cmap='coolwarm'
    )
    plot_thermal_strain_edges(
        node_coords = node_xyz, connectivity=ConnectivityMatrix_line, 
        epsilon_th=Y_array, title="Young's modulus", cmap='coolwarm'
    )
    plot_thermal_strain_edges_CustomRange(
        node_coords = node_xyz, connectivity=ConnectivityMatrix_line, 
        epsilon_th=Y_array, title="Young's modulus (by Yhard)", cmap='coolwarm', 
        vmin=cfg.Yratio * cfg.Yhard, vmax=cfg.Yhard
    )
    plot_thermal_strain_edges_CustomRange(
        node_coords = node_xyz, connectivity=ConnectivityMatrix_line, 
        epsilon_th=Y_array, title="Young's modulus (by Ysoft)", cmap='coolwarm', 
        vmin=cfg.Yratio * cfg.Ysoft, vmax=cfg.Ysoft
    )

    # 3D Reference Plot (Original Figure 4)
    fig, ax = new_fig(4, projection='3d')
    # ... (code for plotting 3D wireframe - cannot be easily injected from geometry.py
    # because it requires looping over Triangles) ...
    # We rely on the call to plot_truss_3d at the end of the simulation.
    


# ==============================================================================
# 4. System Model and Initial State Setup
# ==============================================================================

# 4.1 Boundary Conditions
bc1 = BoundaryConditions3D(Ndofs)

# 4.2 Elastic Model
elastic_model1 = ElasticGHEdgesCoupledThermal(
    energy_choice = 4, 
    Nedges = Nedges, NP_total = NP_total, Ndofs = Ndofs,
    connectivity = ConnectivityMatrix_line, l0_ref = L0,
    ks_array = ks_array, hinge_quads = HingeQuads_order,
    theta_bar = cfg.theta_bar, kb_array = kb_array,
    beta = cfg.beta, epsilon_th = eps_th_vector,
    model_choice = cfg.model_choice
)

# 4.3 Mass and Gravity
NodalMass = 1e-7 
massVector = np.full(Ndofs, NodalMass)

g_vec_raw = cfg.g_vec_default * (-1.0 if cfg.iGravity == 1 else 0.0)
stepper_g_vec = -g_vec_raw # The z axis is inverse of physical world. 05-21-25

# 4.4 Time Stepper Initialization
dt = cfg.dt # Use initial dt
stepper3D = timeStepper3D_static(
    massVector, dt, cfg.qtol, cfg.maxIter,
    stepper_g_vec, bc1, elastic_model1, X0)

# 4.5 History Arrays
Q_history = np.zeros((cfg.n_record, Ndofs))
R_history = np.zeros((cfg.n_record, Ndofs))
strain_history = np.zeros((cfg.n_record, Nedges))
stress_history = np.zeros((cfg.n_record, Nedges))
theta_history = np.zeros((cfg.n_record, n_hinges))
length_history = np.zeros((cfg.n_record, Nedges))

q_old = X0.copy()
u_old = np.zeros(Ndofs) 
a_old = np.zeros(Ndofs) 

# store initial state at step=0
record_step(
    step=0, q_new=q_old, elastic_model=elastic_model1, 
    connectivity=ConnectivityMatrix_line, L0=L0, ks_array=ks_array, 
    hinge_quads=HingeQuads_order, Q_history=Q_history, R_history=R_history, 
    length_history=length_history, strain_history=strain_history, 
    stress_history=stress_history, theta_history=theta_history)

print("---Netwon iteration starts---")


# ==============================================================================
# 5. Main Simulation Loop (Thermal Actuation)
# ==============================================================================

step_log = []
time_log = []
t = 0.0
step = 0

while True:
    
    # Adaptive time step logic
    if t + dt > cfg.totalTime:
        dt = cfg.totalTime - t
    stepper3D.dt = dt
    
    t_next = t + dt
    elastic_model1.eps_th = eps_th_vector_pattern * (t_next / cfg.totalTime)
    
    print(f"t = {t_next:.8f}, g={stepper3D.g}\n") 

    # Boundary Condition Update (Clamping)
    fixedNodes = np.array([], dtype=int)
    if cfg.iMesh == 1:
        fixedNodes = fun_BC_3D_hold_center(bc1, X0, NP_total, half_x=0.01, half_y=0.01)
    elif cfg.iMesh == 2:
        fixedNodes = fun_BC_peanut(bc1, X0, NP_total,
                                   x_min=0.0577, x_max=0.0987,
                                   y_min=0.0458, y_max=0.0538,
                                   node_region_fn=region_fn)
    elif cfg.iMesh == 3:
        temp = 0.01
        fixedNodes = fun_BC_peanut(bc1, X0, NP_total,
                                   x_min=x_center-temp, x_max=x_center+temp,
                                   y_min=y_center-temp, y_max=y_center+temp,
                                   node_region_fn=region_fn)
    
    q_new, converged = stepper3D.simulate(q_old, q_old, u_old, a_old)
    
    if not converged:
        if dt > cfg.dt_min:
            dt = max(dt * 0.5, cfg.dt_min)
            print(f"Newton failed at t={t_next:.8f}, reducing dt to {dt:.8f} and retrying")
            continue
        else:
            print(f"Not converged at t = {t:.8f} s, saving failure state")
            record_step(1, q_new, elastic_model1, ConnectivityMatrix_line, L0, ks_array, HingeQuads_order, Q_history, R_history, length_history, strain_history, stress_history, theta_history)
            break
    
    # Successful Step
    t = t_next
    step += 1
    q_old = q_new.copy()
    step_log.append(step)
    time_log.append(t)
    
    # Adaptive time step increase logic
    if stepper3D.last_num_iters < 5:
        dt = min(dt * 1.05, cfg.dt_max)
        print("increase dt\n") 
    elif abs(t - cfg.totalTime) < 1e-12:
        pass
    else:
        # This conditional newline is vital to match the original output spacing
        print("\n") 

    if abs(t - cfg.totalTime) < 1e-12:
        print("Simulation converged, saving final state")
        record_step(1, q_new, elastic_model1, ConnectivityMatrix_line, L0, ks_array, HingeQuads_order, Q_history, R_history, length_history, strain_history, stress_history, theta_history)
        break

print("--- Thermal actuation completed ---")


# --- Plotting Final Thermal State ---
plot_truss_3d(
    q_old,
    ConnectivityMatrix_line,
    NP_total=NP_total,
    title=f"t = {t:.8f} s",
    show_labels=False)


# --- Save data after thermal actuation step ---
mdict = {
    'X0_4columns': X0_4columns, 'ConnectivityMatrix_line': ConnectivityMatrix_line, 'Triangles': Triangles,
    'NP_total': NP_total, 'Nedges': Nedges, 'Ntriangles': Ntriangles, 'Ndofs': Ndofs, 
    'X0': X0, 'L0': L0, 'ks_array': ks_array, 'kb_array': kb_array, 
    'Q_history': Q_history, 'R_history': R_history, 'strain_history': strain_history, 
    'stress_history': stress_history, 'HingeQuads_order': HingeQuads_order, 
    'theta_history': theta_history, 'time_log': time_log, 'step_log': step_log, 
    'q_old': q_old, 'fixedNodes': fixedNodes,
}
sio.savemat('../../output/output_deps_thermal_WithGravity.mat', mdict, do_compression=True)
print("Data saved at ", ", ".join(mdict.keys()))


# ==============================================================================
# 6. Gravity Ramp-Down Loop
# ==============================================================================

relax_total = 1.0
relax_steps = 10
dt_relax = relax_total / relax_steps
g0 = stepper_g_vec 
extra_records = relax_steps

# Allocate history arrays for ramp-down (appends 1 slot at the end for the final state)
n_current_records = Q_history.shape[0]
Q_history = np.vstack([Q_history, np.zeros((1, Ndofs))])
R_history = np.vstack([R_history, np.zeros((1, Ndofs))])
strain_history = np.vstack([strain_history, np.zeros((1, Nedges))])
stress_history = np.vstack([stress_history, np.zeros((1, Nedges))])
theta_history = np.vstack([theta_history, np.zeros((1, n_hinges))])
length_history = np.vstack([length_history, np.zeros((1, Nedges))])


print("\n=== Gravity ramp‐down ===")
# Note: 'step' from the thermal phase continues here for logging total steps
for k in range(1, relax_steps + 1):
    t_rel = k * dt_relax
    factor = max(0.0, 1.0 - t_rel / relax_total)
    
    stepper3D.g = g0 * factor
    stepper3D.makeWeight()
    
    # Print Gravity Update (matches original output)
    print(f"k = {k:d}, factor={factor:.3f}, g={stepper3D.g} \n")

    # Step forward
    q_new, converged = stepper3D.simulate(q_old, q_old, u_old, a_old)
    
    if not converged:
        raise RuntimeError(f"Failed to converge during gravity‐ramp at step {k}")

    # Advance state and log timing (t is final time after thermal phase)
    q_old = q_new.copy()
    step_log.append(step + k)
    time_log.append(t + t_rel)
    
    # Record final no-gravity state into the last slot (index 2)
    if k == relax_steps:
        record_step(
            n_current_records, 
            q_new, elastic_model1, ConnectivityMatrix_line, L0, ks_array, HingeQuads_order, 
            Q_history, R_history, length_history, strain_history, stress_history, theta_history
        )

print("Gravity ramp‐down completed")

# --- Plotting Final No-Gravity State ---
plot_truss_3d(
    q_old,
    ConnectivityMatrix_line,
    NP_total=NP_total,
    title=f"t = {t + relax_total:.8f} s",
    show_labels=False)


# --- Save Final Data (output_deps_thermal_NoG.mat) ---
mdict.update({
    'Q_history': Q_history, 'R_history': R_history, 'strain_history': strain_history, 
    'stress_history': stress_history, 'theta_history': theta_history,
    'time_log': time_log, 'step_log': step_log, 'q_old': q_old, 
    'extra_records': extra_records
})

sio.savemat('../../output/output_deps_thermal_NoG.mat', mdict, do_compression=True)
print("Data saved at ", ", ".join(mdict.keys()))

end = time.perf_counter()
print(f"Elapsed time: {end - start:.4f} s")