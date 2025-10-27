# src/thermoshell/simulation.py

import numpy as np
import time
from typing import Tuple, Dict, Callable, Any
from functools import partial
from collections import Counter, defaultdict
from itertools import combinations

# --- Imports from thermoshell package ---
from src.thermoshell.geometry.mesh_io import load_mesh
from src.thermoshell.geometry.mesh_props import calculate_edge_lengths
from src.thermoshell.material.bilayer import bilayer_flexural_rigidity
from src.thermoshell.material.assignment import (
    assign_thermal_strains_contour,
    assign_youngs_modulus_v3 # Assuming v3 is the desired function
)
from src.thermoshell.material.fluctuations import add_boundary_fluctuations
# Import specific region functions needed for setup
from src.thermoshell.patterning.regions import (
    circle_six_arms_region,
    square_X_region
)
from src.thermoshell.patterning.complex import whole_peanut_region
from src.thermoshell.assembly.assemblers import ElasticGHEdgesCoupledThermal
from src.thermoshell.solver.boundary_conditions import BoundaryConditions3D
from src.thermoshell.solver.time_stepper import timeStepper3D_static, record_step # Using static solver

# --- Helper Function ---
def _find_hinges_and_order(Triangles, ConnectivityMatrix_line, X0_4columns):
    """Finds hinge edges and determines node order for dihedral angles."""
    # Build a flat list of all undirected edges from the triangle list
    tri_edges = []
    for _, v1, v2, v3 in Triangles.astype(int):
        tri_edges += [
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v3))),
            tuple(sorted((v3, v1))),
        ]

    # Count how many times each edge appears
    edge_counts = Counter(tri_edges)

    # Keep only those edges that appear exactly twice (interior edges = hinges)
    hinge_keys = {edge for edge, cnt in edge_counts.items() if cnt == 2}

    # Extract their IDs from your connectivity table
    hinge_edges = []
    edgeid_to_nodes = {}
    for eid, n0, n1 in ConnectivityMatrix_line.astype(int):
        edgeid_to_nodes[eid] = (n0, n1)
        if tuple(sorted((n0, n1))) in hinge_keys:
            hinge_edges.append(eid)

    hinge_edges = np.array(hinge_edges, dtype=int)

    # --- Find opposite vertices ---
    edge_to_opps = defaultdict(list)
    for _, v1, v2, v3 in Triangles.astype(int):
        verts = (v1, v2, v3)
        for a, b in combinations(verts, 2):
            key = tuple(sorted((a, b)))
            opp = next(v for v in verts if v not in key)
            edge_to_opps[key].append(opp)

    # --- Collect and Order Hinge Quads ---
    HingeQuads = []
    for eid in hinge_edges:
        n0, n1 = edgeid_to_nodes[eid]
        key = tuple(sorted((n0, n1)))
        if key in edge_to_opps and len(edge_to_opps[key]) == 2:
             oppA, oppB = edge_to_opps[key]
             HingeQuads.append([eid, n0, n1, oppA, oppB])
        # else: # Handle boundary or malformed edges if necessary
        #    print(f"Warning: Edge {eid} identified as hinge but lacks two opposite vertices.")
            
    HingeQuads = np.array(HingeQuads, dtype=int)

    # --- Determine Node Order based on Normals ---
    HingeQuads_order = []
    for eid, n0, n1, oppA, oppB in HingeQuads:
        x0 = X0_4columns[n0, 1:4]
        x1 = X0_4columns[n1, 1:4]
        x2 = X0_4columns[oppA, 1:4]
        x3 = X0_4columns[oppB, 1:4]
        m_e0 = x1 - x0
        m_e1 = x2 - x0
        m_e2 = x3 - x0
        n0_v = np.cross(m_e0, m_e1)
        n1_v = np.cross(m_e2, m_e0)

        # Consistent ordering based on normal directions (adjust logic if needed)
        # This assumes normals should generally point "outward" (positive z)
        if (n0_v[2] < 0) and (n1_v[2] < 0): # Both pointing -z, swap opp vertices
            row = [eid, n0, n1, oppB, oppA]
        elif (n0_v[2] > 0) and (n1_v[2] > 0): # Both pointing +z, keep order
             row = [eid, n0, n1, oppA, oppB]
        elif (n0_v[2] * n1_v[2] < 0):
             # Normals point opposite ways - might indicate complex geometry or ordering issue
             # Defaulting to original order, but might need refinement for specific cases
             print(f"Warning: Hinge edge {eid} has opposing normals. Check mesh consistency.")
             row = [eid, n0, n1, oppA, oppB]
        else: # One or both normals are zero (degenerate triangle?)
             print(f"Warning: Hinge edge {eid} involves degenerate triangle normal. Using default order.")
             row = [eid, n0, n1, oppA, oppB]

        HingeQuads_order.append(row)

    return np.array(HingeQuads_order, dtype=int), hinge_edges

# --- Main Simulation Functions ---

def setup_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads mesh, calculates properties, initializes models based on config.
    """
    print("--- Setting up simulation ---")
    start_setup = time.perf_counter()

    # 1. Load Mesh
    mesh_file = config['mesh_file']
    X0_4columns, ConnectivityMatrix_line, Triangles = load_mesh(mesh_file)
    print(f"Loaded mesh: {mesh_file}")

    NP_total = X0_4columns.shape[0]
    Nedges = ConnectivityMatrix_line.shape[0]
    Ndofs = NP_total * 3
    X0 = X0_4columns[:, 1:4].ravel() # Initial coordinates (flat)
    node_xyz = X0_4columns[:, 1:4] # Initial coordinates (Nx3)

    # 2. Calculate Mesh Props & Hinges
    L0 = calculate_edge_lengths(X0_4columns, ConnectivityMatrix_line)
    HingeQuads_order, hinge_eids = _find_hinges_and_order(Triangles, ConnectivityMatrix_line, X0_4columns)
    n_hinges = HingeQuads_order.shape[0]
    print(f"Nodes: {NP_total}, Edges: {Nedges}, Hinges: {n_hinges}, DOFs: {Ndofs}")

    # 3. Calculate Material/Stiffness Properties
    Ysoft = config['Ysoft']
    Yhard = config['Yhard']
    lk, h1, h2 = config['lk'], config['h1'], config['h2']

    D_12 = bilayer_flexural_rigidity(h1, h2, Ysoft, Yhard) # Assuming width b=1
    kb1  = (2.0 / np.sqrt(3.0)) * (Ysoft * (h1**3) / 12.0)
    kb12 = (2.0 / np.sqrt(3.0)) * D_12
    ks1  = Ysoft * h1 * (lk**2) * np.sqrt(3) / 2.0
    ks2  = Yhard * h2 * (lk**2) * np.sqrt(3) / 2.0
    ks12 = ks1 + ks2

    # 4. Assign Patterned Properties (Y_array, eps_th_vector)
    #    Determine region function based on config
    pattern_type = config.get('pattern_type', None)
    region_fn = None
    region_fn_lookup = {} # Store for potential use in BCs

    mesh_center_x = 0.5 * (node_xyz[:, 0].min() + node_xyz[:, 0].max())
    mesh_center_y = 0.5 * (node_xyz[:, 1].min() + node_xyz[:, 1].max())
    calculated_center = (mesh_center_x, mesh_center_y)

    if pattern_type == 'circle_six_arms':
        pattern_center = config.get('Pattern_center', calculated_center) # Use config or calculated
        region_fn = partial(
            circle_six_arms_region,
            circle_center=pattern_center,
            circle_radius=config['Stripe_r'],
            arm_half_width=config['StripeWidth'],
            arm_half_length=config['StripeLength']
        )
    elif pattern_type == 'square_X':
        pattern_center = config.get('Pattern_center', calculated_center)
        region_fn = partial(
            square_X_region,
            circle_center=pattern_center,
            circle_radius=config['Stripe_r'],
            arm_half_width=config['StripeWidth'],
            arm_half_length=config['StripeLength']
        )
    elif pattern_type == 'whole_peanut':
        pattern_center = config.get('Pattern_center', calculated_center)
        region_fn = partial(
            whole_peanut_region,
            center=pattern_center,
            delta_shape=config['delta_shape'],
            star_radius=config['star_radius'],
            star_thickness=config['star_thickness'],
            n_spokes=config['n_spokes'],
            beam_thickness=config['beam_thickness']
        )
    # Add elif for other pattern_types...
    else:
        print(f"Warning: No valid pattern_type specified ('{pattern_type}'). "
              "Material properties will be uniform.")
        # Default: Assign uniform properties if no pattern specified
        Y_array = np.full(Nedges, Ysoft) # Default to Ysoft everywhere
        eps_th_vector = np.zeros(Nedges) # Default to zero thermal strain

    if region_fn:
        # Assign Young's Modulus based on pattern
        # Note: Using v3 which includes thresholding logic; adjust if needed
        Y_override = config.get('Y_override_params', {}) # Get override dict or empty
        Y_array = assign_youngs_modulus_v3(
            node_xyz, ConnectivityMatrix_line, region_fn=region_fn,
            circle_center=pattern_center, # Requires pattern_center to be defined
            circle_radius=config['OuterR'],
            Ysoft=Ysoft, Yhard=Yhard, Yratio=config['Yratio'],
            inside=False, # Assuming pattern defines 'hard' region, soft is outside
            **Y_override # Pass threshold params directly
        )

        # Assign Thermal Strain based on pattern
        eps_th_vector = assign_thermal_strains_contour(
            node_xyz, ConnectivityMatrix_line, config['eps_thermal'],
            region_fn=region_fn, inside=False # Strain assigned outside the pattern
        )

        # Apply fluctuations if enabled
        if config.get('apply_fluctuations', False):
            print("Applying boundary fluctuations to thermal strain.")
            eps_th_vector = add_boundary_fluctuations(
                node_xyz, ConnectivityMatrix_line, eps_th_vector,
                amp=config['eps_thermal'] * config.get('fluctuation_amp_factor', 5.0), # Example factor
                n_waves=config.get('fluctuation_n_waves', 0.5),
                decay_width=config.get('fluctuation_decay', 0.03)
            )
        # Store the region function if needed later (e.g., by BCs)
        region_fn_lookup[config['mesh_id']] = region_fn

    # 5. Calculate Final Element Stiffness Arrays
    # Map Y_array to ks_array based on bilayer logic
    ks_array = np.zeros(Nedges)
    ks_array[Y_array == Yhard] = ks12
    ks_array[Y_array == Ysoft] = ks1
    # Handle intermediate values if Y_array includes ramps - requires interpolation logic
    if not np.all((Y_array == Yhard) | (Y_array == Ysoft)):
         print("Warning: Y_array contains values other than Yhard/Ysoft. "
               "Using simple mapping for ks_array - refine if ramps are used.")

    # Map Y_array (at hinges) to kb_array
    Y_array_hinges = Y_array[hinge_eids]
    kb_array = np.zeros(n_hinges)
    kb_array[Y_array_hinges == Yhard] = kb12
    kb_array[Y_array_hinges == Ysoft] = kb1
    if not np.all((Y_array_hinges == Yhard) | (Y_array_hinges == Ysoft)):
        print("Warning: Y_array at hinges contains intermediate values. "
              "Using simple mapping for kb_array - refine if ramps are used.")

    # Apply global scaling factors
    ks_array *= config['FactorKs']
    kb_array *= config['FactorKb']

    # 6. Initialize Models
    bc_object = BoundaryConditions3D(Ndofs)

    # Use the thermal coupled model
    elastic_model = ElasticGHEdgesCoupledThermal(
        energy_choice=4, # Coupled bending + stretch
        Nedges=Nedges, NP_total=NP_total, Ndofs=Ndofs,
        connectivity=ConnectivityMatrix_line, l0_ref=L0,
        ks_array=ks_array, hinge_quads=HingeQuads_order,
        theta_bar=config['theta_bar'], kb_array=kb_array,
        beta=config['beta'],
        epsilon_th=np.zeros_like(eps_th_vector), # Start with zero thermal strain
        model_choice=1 # Assuming analytical model
    )
    # Store the target thermal strain vector separately for ramping
    target_eps_th_vector = eps_th_vector.copy()

    # Initialize Stepper (Static Solver)
    massVector = np.full(Ndofs, config['nodal_mass'])
    gravity_vec = config['gravity_vec'] if config['gravity_on'] else np.zeros(3)
    # Handle sign convention if needed (e.g., if Z points up in mesh but gravity is -Z)
    # Example: gravity_vec[2] *= -1 # If mesh Z is opposite to physical Z
    
    stepper = timeStepper3D_static(
        massVector, config['dt_init'], config['qtol'], config['maxIter'],
        gravity_vec, bc_object, elastic_model, X0
    )

    # 7. Package Initial State and Objects
    initial_state = {
        'q_old': X0.copy(),
        'u_old': np.zeros(Ndofs),
        'a_old': np.zeros(Ndofs),
        'X0': X0.copy()
    }
    mesh_data = {
        'X0_4columns': X0_4columns,
        'ConnectivityMatrix_line': ConnectivityMatrix_line,
        'Triangles': Triangles,
        'NP_total': NP_total,
        'Nedges': Nedges,
        'Ndofs': Ndofs,
        'L0': L0,
    }
    hinge_data = {
        'HingeQuads_order': HingeQuads_order,
        'n_hinges': n_hinges
    }
    # Data to save (excluding large arrays passed separately)
    save_data = {
         'config': config, # Save the config used
         # Add other small metadata as needed
    }


    end_setup = time.perf_counter()
    print(f"--- Setup complete ({end_setup - start_setup:.4f} s) ---")

    return {
        "stepper": stepper,
        "elastic_model": elastic_model,
        "bc_object": bc_object,
        "initial_state": initial_state,
        "mesh_data": mesh_data,
        "hinge_data": hinge_data,
        "target_eps_th_vector": target_eps_th_vector,
        "region_fn_lookup": region_fn_lookup, # Pass back for BCs if needed
        "save_data": save_data # Base data for saving
    }


def run_quasi_static_actuation(stepper, elastic_model, bc_apply_func: Callable,
                               sim_params: dict, initial_state: dict,
                               mesh_data: dict, hinge_data: dict,
                               target_eps_th_vector: np.ndarray) -> Dict[str, Any]:
    """
    Runs the quasi-static actuation phase, ramping thermal strain.
    """
    print("--- Running Quasi-Static Actuation ---")
    start_run = time.perf_counter()

    # Extract parameters
    totalTime = sim_params['totalTime']
    dt_min = sim_params['dt_min']
    dt_max = sim_params['dt_max']
    dt = sim_params.get('dt_init', dt_max * 0.1) # Use initial or default
    n_record = sim_params.get('n_record', 2) # Default to start/end

    # Extract state and data
    q_old = initial_state['q_old'].copy()
    u_old = initial_state['u_old'].copy()
    a_old = initial_state['a_old'].copy()
    X0 = initial_state['X0']

    Ndofs = mesh_data['Ndofs']
    Nedges = mesh_data['Nedges']
    n_hinges = hinge_data['n_hinges']

    # Initialize History (adjust size if n_record > 2)
    Q_history = np.zeros((n_record, Ndofs))
    R_history = np.zeros((n_record, Ndofs))
    strain_history = np.zeros((n_record, Nedges))
    stress_history = np.zeros((n_record, Nedges))
    theta_history = np.zeros((n_record, n_hinges))
    length_history = np.zeros((n_record, Nedges))

    # Record initial state (step 0)
    record_step(
        step=0, q_new=q_old, elastic_model=elastic_model,
        connectivity=mesh_data['ConnectivityMatrix_line'], L0=mesh_data['L0'],
        ks_array=elastic_model.ks_array, # Get current ks from model
        hinge_quads=hinge_data['HingeQuads_order'],
        Q_history=Q_history, R_history=R_history, length_history=length_history,
        strain_history=strain_history, stress_history=stress_history,
        theta_history=theta_history
    )

    # --- Main Loop ---
    step_log = [0]
    time_log = [0.0]
    t = 0.0
    step_counter = 0 # Actual number of steps taken

    while abs(t - totalTime) > 1e-12:
        step_counter += 1
        # Determine current dt, ensuring we don't overshoot totalTime
        current_dt = min(dt, totalTime - t)
        if abs(current_dt - dt) > 1e-12: # Check if we adjusted dt for the last step
            print(f"Adjusting final dt to {current_dt:.4g}")
        stepper.dt = current_dt # Set dt in stepper if it uses it
        t_next = t + current_dt

        # Update thermal strain based on time (linear ramp)
        elastic_model.eps_th = target_eps_th_vector * (t_next / totalTime)
        
        # Apply Boundary Conditions for this step/time
        # Assumes bc_apply_func takes (bc_object, time, X0, Nnodes)
        fixed_node_ids = bc_apply_func(stepper.bc, t_next, X0, mesh_data['NP_total'])
        if step_counter == 1 and sim_params.get('do_print', False): # Get from sim_params
            fixedVals, fixedIdxs, freeIdxs = stepper.bc.getBoundaryConditions()
            print("Applied BCs:")
            print("  Fixed DOF indices: ", fixedIdxs)
            # print("  Fixed DOF values: ", *("0.0" if v == 0.0 else f"{v:.4e}" for v in fixedVals))
            print("  Fixed node IDs: ", fixed_node_ids)


        # Simulate one step
        q_new, converged = stepper.simulate(q_old, q_old, u_old, a_old)

        # Handle convergence and adaptive stepping
        if not converged:
            if dt > dt_min:
                dt = max(dt * 0.5, dt_min)
                print(f"Newton failed at t={t_next:.4g}, reducing dt to {dt:.4g} and retrying step.")
                # Reset time step in stepper if necessary
                stepper.dt = dt # May not be needed if simulate doesn't use it directly
                # DO NOT advance time or state, loop will retry with smaller dt
                continue
            else:
                print(f"ERROR: Convergence failed at minimum dt ({dt_min:.2g}) at t ~ {t_next:.4g}. Aborting.")
                # Record failure state?
                record_step(1, q_new, elastic_model, mesh_data['ConnectivityMatrix_line'],
                            mesh_data['L0'], elastic_model.ks_array, hinge_data['HingeQuads_order'],
                            Q_history, R_history, length_history, strain_history,
                            stress_history, theta_history)
                success = False
                break # Exit the while loop on failure
        else:
             # Successful step
             t = t_next
             q_old = q_new.copy() # Update state for next step
             step_log.append(step_counter)
             time_log.append(t)
             print(f"Step {step_counter} converged (t={t:.4g}, iters={stepper.last_num_iters}, dt={current_dt:.4g})")

             # Adapt dt for next step (increase if converged quickly)
             if stepper.last_num_iters < 5 and abs(t - totalTime) > 1e-12: # Don't increase on last step
                 dt = min(dt * 1.05, dt_max)
                 # print(f"  Increasing dt to {dt:.4g}")
             success = True # Mark simulation as successful so far


    # Record final state (step 1 in history array if n_record=2)
    if success:
        print(f"Actuation simulation converged at t = {t:.4g}")
        record_step(n_record - 1, q_new, elastic_model, mesh_data['ConnectivityMatrix_line'],
                     mesh_data['L0'], elastic_model.ks_array, hinge_data['HingeQuads_order'],
                     Q_history, R_history, length_history, strain_history,
                     stress_history, theta_history)

    end_run = time.perf_counter()
    print(f"--- Actuation run complete ({end_run - start_run:.4f} s) ---")

    return {
        "success": success,
        "Q_history": Q_history,
        "R_history": R_history,
        "strain_history": strain_history,
        "stress_history": stress_history,
        "theta_history": theta_history,
        "length_history": length_history,
        "time_log": np.array(time_log),
        "step_log": np.array(step_log),
        "q_final": q_new, # Final converged state
        "n_records": n_record # Number of records in history arrays
        # Add other relevant outputs as needed
    }

def run_relaxation(stepper, elastic_model, bc_apply_func: Callable,
                   relax_params: dict, current_state: dict,
                   mesh_data: dict, hinge_data: dict,
                   n_record_offset: int = 0) -> Dict[str, Any]:
    """
    Runs the relaxation phase, ramping down gravity.
    Appends results to existing history arrays if provided.
    """
    print("--- Running Relaxation Phase ---")
    start_run = time.perf_counter()

    # Extract parameters
    relax_total_time = relax_params['relax_total_time']
    relax_steps = relax_params['relax_steps']
    g_initial = relax_params.get('g_initial', stepper.g) # Use initial g or current stepper g

    dt_relax = relax_total_time / relax_steps

    # Extract state and data
    q_old = current_state['q_old'].copy()
    u_old = current_state['u_old'].copy() # Usually zero for static start
    a_old = current_state['a_old'].copy() # Usually zero
    X0 = current_state['X0'] # Need X0 for BCs

    # History arrays (passed in for appending or create new)
    # This example assumes we record *only* the final relaxation state
    # If full history needed, initialize/extend arrays here.
    n_record = 1 # Just the final state for this example

    Q_relax_hist = np.zeros((n_record, mesh_data['Ndofs']))
    # ... initialize other history arrays if needed ...

    # --- Relaxation Loop ---
    time_offset = relax_params.get('time_offset', 0.0) # Time at start of relaxation
    step_offset = relax_params.get('step_offset', 0)
    relax_time_log = []
    relax_step_log = []

    for k in range(1, relax_steps + 1):
        t_rel = k * dt_relax
        current_time = time_offset + t_rel

        # Update gravity
        factor = max(0.0, 1.0 - t_rel / relax_total_time)
        stepper.g = g_initial * factor
        stepper.makeWeight() # Recalculate Fg based on new gravity

        # Apply boundary conditions (using the same function as actuation)
        fixed_node_ids = bc_apply_func(stepper.bc, current_time, X0, mesh_data['NP_total'])

        # Simulate one step (using static solver's simulate)
        q_new, converged = stepper.simulate(q_old, q_old, u_old, a_old) # Pass q_old as guess

        if not converged:
             # Static solver might not need dt reduction, just report failure
             print(f"ERROR: Relaxation step {k} failed to converge (t={current_time:.4g}). Aborting.")
             success = False
             # Optionally record failure state
             if n_record > 0:
                  record_step(0, q_new, elastic_model, mesh_data['ConnectivityMatrix_line'],
                              mesh_data['L0'], elastic_model.ks_array, hinge_data['HingeQuads_order'],
                              Q_relax_hist, # Pass relax-specific histories
                              np.zeros_like(Q_relax_hist), np.zeros((n_record, mesh_data['Nedges'])), # Placeholder R, strain etc.
                              np.zeros((n_record, mesh_data['Nedges'])), np.zeros((n_record, hinge_data['n_hinges'])),
                              np.zeros((n_record, mesh_data['Nedges'])))

             break # Exit loop
        else:
             print(f"Relax step {k} converged (t={current_time:.4g}, factor={factor:.3f}, iters={stepper.last_num_iters})")
             q_old = q_new.copy() # Update for next step
             relax_step_log.append(step_offset + k)
             relax_time_log.append(current_time)
             success = True

    # Record final state
    if success and n_record > 0:
         print(f"Relaxation simulation converged at t = {current_time:.4g}")
         record_step(n_record - 1, q_new, elastic_model, mesh_data['ConnectivityMatrix_line'],
                     mesh_data['L0'], elastic_model.ks_array, hinge_data['HingeQuads_order'],
                     Q_relax_hist, # Use relaxation-specific history
                     np.zeros_like(Q_relax_hist), np.zeros((n_record, mesh_data['Nedges'])), # Placeholder R, strain etc.
                     np.zeros((n_record, mesh_data['Nedges'])), np.zeros((n_record, hinge_data['n_hinges'])),
                     np.zeros((n_record, mesh_data['Nedges'])))


    end_run = time.perf_counter()
    print(f"--- Relaxation run complete ({end_run - start_run:.4f} s) ---")

    return {
        "success": success,
        "q_final_relax": q_new,
        "relax_time_log": np.array(relax_time_log),
        "relax_step_log": np.array(relax_step_log),
        # Return history arrays if they were populated
        "Q_history_relax": Q_relax_hist if n_record > 0 else None,
        # ... R_history_relax, etc. ...
    }