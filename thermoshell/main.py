import numpy as np
import time
import scipy.io as sio
import dataclasses
from typing import Dict, Any

# --- Core and Utilities
from thermoshell.core.utils import parse_args, calculate_pattern_center
from thermoshell.core.params import GeomParams, MaterialParams, SimParams, DerivedParams

# --- Geometry and Connectivity
from thermoshell.geometry.mesh_loader import load_mesh, mesh_files
from thermoshell.geometry.connectivity import get_hinge_connectivity, get_edge_map
from thermoshell.geometry.analysis_helpers import fun_edge_lengths

# --- Materials and Stiffness
from thermoshell.materials.properties import derive_bilayer_constants
from thermoshell.materials.spatial_assignment import assign_spatial_properties, initialize_stiffness_arrays

# --- Loading and Forces
from thermoshell.loading.thermal_generation import generate_thermal_load_functions, apply_thermal_strain_pattern
from thermoshell.loading.boundary_conditions import BoundaryConditions3D, apply_boundary_conditions
from thermoshell.external_forces.gravity import get_gravity_force_vector

# --- Dynamics and Solver
from thermoshell.dynamics.elastic_model import ElasticGHEdgesCoupledThermal
from thermoshell.solver.quasi_static_newton import timeStepper3D_static
from thermoshell.solver.recorders import initialize_history_arrays, record_step
from thermoshell.visualization.plotter import plot_truss_3d, plot_thermal_strain_edges


def run_simulation() -> None:
    """
    Main function to load parameters, assemble the FE model, and run the simulation loops.
    """
    start_time = time.perf_counter()

    # --- 1. INITIAL SETUP AND PARAMETER OVERRIDES ---
    
    # Load default parameters
    P_GEOM = GeomParams()
    P_MAT = MaterialParams()
    P_SIM = SimParams()
    
    # Override defaults using command-line arguments (from argparse)
    args = parse_args()
    
    # Use dataclasses.replace to create new instances with overridden values
    if args.mesh is not None:
        P_GEOM = dataclasses.replace(P_GEOM, mesh_id=args.mesh)
    if args.eps_thermal is not None:
        P_MAT = dataclasses.replace(P_MAT, thermal_strain_mag=args.eps_thermal)
    if args.gravity is not None:
        P_SIM = dataclasses.replace(P_SIM, use_gravity=bool(args.gravity))
    if args.fluctuate is not None:
        P_SIM = dataclasses.replace(P_SIM, use_fluctuations=bool(args.fluctuate))
    if args.do_print:
        P_SIM = dataclasses.replace(P_SIM, do_print=True)
    if args.do_plot:
        P_SIM = dataclasses.replace(P_SIM, do_plot=True)
    if args.do_test:
        P_SIM = dataclasses.replace(P_SIM, do_test=True)

    if P_SIM.do_print:
        print("--- Simulation Configuration ---")
        print(P_GEOM)
        print(P_MAT)
        print(P_SIM)
        print("-" * 30)


    # --- 2. LOAD MESH AND DERIVE CONNECTIVITY ---
    
    try:
        mesh_file = mesh_files[P_GEOM.mesh_id]
    except KeyError:
        print(f"Error: Invalid mesh_id {P_GEOM.mesh_id}. Valid options are: {list(mesh_files.keys())}")
        return

    X0_4columns, ConnectivityMatrix_line, Triangles = load_mesh(mesh_file)

    # Extract core geometry data
    NP_total = X0_4columns.shape[0]
    Nedges = ConnectivityMatrix_line.shape[0]
    X0 = X0_4columns[:, 1:4].ravel()
    Ndofs = X0.size
    node_xyz = X0_4columns[:, 1:4]

    # Calculate reference lengths
    L0 = fun_edge_lengths(X0_4columns, ConnectivityMatrix_line)

    # Derive hinge connectivity (HingeQuads_order) and edge map (edge_dict)
    HingeQuads_order, edge_dict = get_hinge_connectivity(X0_4columns, ConnectivityMatrix_line, Triangles)
    n_hinges = HingeQuads_order.shape[0]

    # Recalculate pattern center based on loaded mesh bounds
    x_center, y_center = calculate_pattern_center(node_xyz)
    P_GEOM = dataclasses.replace(P_GEOM, pattern_center_x=x_center, pattern_center_y=y_center)


    # --- 3. MATERIAL AND LOADING ASSIGNMENT ---

    # Get effective material constants (lk, h_soft, h_hard) for the current mesh
    P_GEOM = derive_bilayer_constants(P_GEOM)

    # Generate region function and patterned thermal strain
    region_fn, bc_fn = generate_thermal_load_functions(P_GEOM)
    eps_th_patterned = apply_thermal_strain_pattern(P_MAT, P_GEOM, node_xyz, ConnectivityMatrix_line, region_fn)

    # Visualization of initial properties
    if P_SIM.do_plot:
        plot_thermal_strain_edges(node_coords=node_xyz, connectivity=ConnectivityMatrix_line,
                                  epsilon_th=eps_th_patterned, title="Initial Thermal Strain Pattern ($\epsilon^{th}$)",
                                  cmap='coolwarm', vmin=P_MAT.thermal_strain_mag, vmax=0.0)

    # Assign spatially varying Young's Modulus and initialize ks/kb arrays
    Y_array = assign_spatial_properties(P_MAT, P_GEOM, node_xyz, ConnectivityMatrix_line, region_fn)
    ks_array, kb_array = initialize_stiffness_arrays(P_MAT, P_GEOM, Y_array, HingeQuads_order)
    
    if P_SIM.do_plot:
        plot_thermal_strain_edges(node_coords=node_xyz, connectivity=ConnectivityMatrix_line,
                                  epsilon_th=Y_array, title="Initial Young's Modulus (Y)",
                                  cmap='cividis')

    # Gravity force vector
    g_vec_nominal = P_SIM.gravity_vector
    if P_SIM.use_gravity:
        mass_vector = np.full(Ndofs, P_MAT.nodal_mass)
        fg_vector = get_gravity_force_vector(mass_vector, g_vec_nominal)
    else:
        fg_vector = np.zeros(Ndofs)


    # --- 4. ASSEMBLE ELASTIC MODEL AND SOLVER ---

    # The full elastic/thermo-mechanical model
    elastic_model = ElasticGHEdgesCoupledThermal(
        energy_choice=4,
        Nedges=Nedges,
        NP_total=NP_total,
        Ndofs=Ndofs,
        connectivity=ConnectivityMatrix_line,
        l0_ref=L0,
        ks_array=ks_array,
        hinge_quads=HingeQuads_order,
        theta_bar=0.0,
        kb_array=kb_array,
        beta=P_MAT.coupling_beta,
        epsilon_th=eps_th_patterned * 0.0, # Start with zero strain
        edge_dict=edge_dict
    )
    
    # Initialize boundary conditions and stepper
    bc = BoundaryConditions3D(Ndofs)

    stepper = timeStepper3D_static(
        mass_vector=mass_vector, 
        dt=P_SIM.dt_initial, 
        qtol=P_SIM.tolerance, 
        maxIter=P_SIM.max_newton_iters,
        Fg=fg_vector, # Pass the full gravity force vector directly
        boundaryCondition=bc, 
        elasticModel=elastic_model, 
        X0=X0)


    # --- 5. INITIALIZE STATE AND HISTORY ---
    q_old = X0.copy()
    Q_history, R_history, length_history, strain_history, stress_history, theta_history = \
        initialize_history_arrays(P_SIM.num_records, Ndofs, Nedges, n_hinges)

    # Record initial state (t=0, all loads are zero)
    record_step(0, q_old, elastic_model, ConnectivityMatrix_line, L0, ks_array, HingeQuads_order,
                Q_history, R_history, length_history, strain_history, stress_history, theta_history)

    step_log = []
    time_log = []
    t = 0.0
    dt = P_SIM.dt_initial
    converged = True
    
    if P_SIM.do_print:
        print("\n" + "="*35)
        print("  6. THERMAL ACTUATION PHASE START ")
        print("="*35)
    
    # --- 6. MAIN SIMULATION LOOP (THERMAL RAMP-UP) ---
    while t < P_SIM.total_time:
        t_prev = t
        if t + dt > P_SIM.total_time:
            dt = P_SIM.total_time - t
        stepper.dt = dt
        t_next = t + dt
        load_factor = t_next / P_SIM.total_time

        # 6a. Apply boundary conditions (clamped region)
        fixedNodes = apply_boundary_conditions(bc, X0, NP_total, bc_fn, P_GEOM)
        
        # 6b. Scale thermal strain based on load factor
        elastic_model.eps_th = eps_th_patterned * load_factor

        # 6c. Simulate the step
        q_new, converged = stepper.simulate(q_old)

        if not converged:
            if dt > P_SIM.dt_min:
                # Reduce timestep and retry
                dt = max(dt * 0.5, P_SIM.dt_min)
                if P_SIM.do_print:
                    print(f"t={t_next:.8f} FAILED. Reducing dt to {dt:.8f} and retrying.")
                continue
            else:
                print(f"FATAL: Solver failed to converge at t = {t_prev:.8f}. Aborting.")
                break

        # 6d. Update state and adapt time step
        t = t_next
        step += 1
        q_old = q_new.copy()
        step_log.append(step)
        time_log.append(t)

        # Adaptive time step logic
        if stepper.last_num_iters < 5:
            dt = min(dt * 1.05, P_SIM.dt_max)
        
        if P_SIM.do_print and step % 10 == 0:
             print(f"Step {step:03d} | t={t:.4f} | dt={dt:.2e} | Iters={stepper.last_num_iters}")
        
        # Check for end of ramp-up phase
        if abs(t - P_SIM.total_time) < P_SIM.dt_min:
            if P_SIM.do_print:
                print(f"\nActuation converged in {step} steps. Saving final state (Step 1).")
            record_step(1, q_new, elastic_model, ConnectivityMatrix_line, L0, ks_array, HingeQuads_order,
                        Q_history, R_history, length_history, strain_history, stress_history, theta_history)
            break

    # --- 7. GRAVITY RAMP-DOWN PHASE (RELAXATION) ---

    q_final_actuation = q_old.copy() # Store the state reached at t=total_time

    if P_SIM.use_gravity and converged:
        if P_SIM.do_print:
            print("\n" + "="*35)
            print("  7. GRAVITY RAMP-DOWN PHASE START ")
            print("="*35)

        dt_relax = P_SIM.relax_time / P_SIM.relax_steps
        
        # Ensure thermal load remains constant at max value
        elastic_model.eps_th = eps_th_patterned * P_SIM.total_time

        # Ramp down loop
        for k in range(1, P_SIM.relax_steps + 1):
            t_rel = k * dt_relax
            
            # Linear ramp-down factor: 1.0 -> 0.0
            gravity_factor = max(0.0, 1.0 - t_rel / P_SIM.relax_time)
            
            # Scale the gravity force vector
            stepper.Fg = get_gravity_force_vector(mass_vector, g_vec_nominal) * gravity_factor
            
            # Simulate the relaxation step
            q_new, converged_relax = stepper.simulate(q_old)
            
            if not converged_relax:
                print(f"FATAL: Failed to converge during gravity-ramp at step {k}. Saving previous converged state.")
                break
                
            q_old = q_new.copy()
            
            if P_SIM.do_print and k % 5 == 0:
                 print(f"Relax Step {k:02d} | Grav Factor={gravity_factor:.4f}")

        if converged_relax:
            if P_SIM.do_print:
                print("Gravity ramp-down completed. Saving final state (Step 2, No Gravity).")
            
            # Record final, gravity-free shape
            record_step(2, q_new, elastic_model, ConnectivityMatrix_line, L0, ks_array, HingeQuads_order,
                        Q_history, R_history, length_history, strain_history, stress_history, theta_history)
            time_log.append(t + P_SIM.relax_time)
            step_log.append(step + P_SIM.relax_steps)
            q_final = q_new
        else:
            q_final = q_final_actuation # Use the state before ramp down failed
    else:
        q_final = q_old # Final state is the result of the ramp-up


    # --- 8. VISUALIZATION AND FINAL OUTPUT ---

    if P_SIM.do_plot:
        plot_truss_3d(q_final, ConnectivityMatrix_line, NP_total=NP_total,
                      title=f"Final Deformed Shape (t={t:.4f}s)", show_labels=False)

    # Prepare data dictionary for MATLAB saving
    mdict: Dict[str, Any] = {
        'P_GEOM': dataclasses.asdict(P_GEOM),
        'P_MAT': dataclasses.asdict(P_MAT),
        'P_SIM': dataclasses.asdict(P_SIM),
        'X0_4columns': X0_4columns,
        'ConnectivityMatrix_line': ConnectivityMatrix_line,
        'Triangles': Triangles,
        'X0': X0,
        'L0': L0,
        'ks_array': ks_array,
        'kb_array': kb_array,
        'Q_history': Q_history,
        'R_history': R_history,
        'strain_history': strain_history,
        'stress_history': stress_history,
        'HingeQuads_order': HingeQuads_order,
        'theta_history': theta_history,
        'time_log': np.array(time_log),
        'step_log': np.array(step_log),
        'q_final': q_final,
        'fixedNodes': fixedNodes.astype(np.int32)
    }

    sio.savemat('output_thermoshell_sim.mat', mdict, do_compression=True)
    
    end_time = time.perf_counter()
    if P_SIM.do_print:
        print(f"\nTotal Elapsed Time: {end_time - start_time:.4f} s")
        print("Data successfully saved to output_thermoshell_sim.mat.")


if __name__ == "__main__":
    run_simulation()
