# examples/run_thermoshell_sim.py 
# (Or place it at the root level, but outside src/thermoshell)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import time
import os
from functools import partial 

# --- Imports from the thermoshell package ---
# Assuming 'src' is in your Python path or you run from the directory containing 'src'
from src.thermoshell.config import get_default_config
from src.thermoshell.simulation import setup_simulation, run_quasi_static_actuation, run_relaxation
from src.thermoshell.utils import save_results
# Import specific BC functions needed for THIS example
from src.thermoshell.solver.boundary_conditions import fun_BC_3D_hold_center, fun_BC_peanut 
# Import plotting functions 
from src.thermoshell.viz.mesh_plots import plot_truss_3d
# Import region functions if needed for BCs or setup validation
from src.thermoshell.analysis.patterning.regions import (
    circle_six_arms_region, 
    square_X_region
) 
from src.thermoshell.analysis.patterning.complex import whole_peanut_region

def define_config(args):
    """Creates the configuration dictionary based on defaults and CLI arguments."""
    config = get_default_config() 
    
    # --- Apply overrides from args ---
    if args.mesh is not None:
         config['mesh_id'] = args.mesh
         # --- IMPORTANT: Update dependent defaults ---
         mesh_id = config['mesh_id']
         # Check if mesh_id is valid
         if mesh_id not in config['mesh_files']:
             raise ValueError(f"Mesh ID {mesh_id} not defined in config['mesh_files']")
         if mesh_id not in config['bilayer_params']:
             # Use default if specific params not found, or raise error
             print(f"Warning: Mesh ID {mesh_id} not found in config['bilayer_params']. Using defaults for lk, h1, h2.")
             # Or: raise ValueError(f"Mesh ID {mesh_id} not defined in config['bilayer_params']")
         if mesh_id not in config['pattern_params']:
              print(f"Warning: Mesh ID {mesh_id} not found in config['pattern_params']. Using default pattern params.")
             # Or: raise ValueError(f"Mesh ID {mesh_id} not defined in config['pattern_params']")

         config['mesh_file'] = config['mesh_files'].get(mesh_id, config['mesh_file']) # Fallback to default if somehow missing
         # Update with mesh-specific params, falling back to empty dict if key missing
         config.update(config['bilayer_params'].get(mesh_id, {})) 
         config.update(config['pattern_params'].get(mesh_id, {})) 
         config['output_file_base'] = f"output_mesh{mesh_id}" 
         # --- End update ---
         
    if args.eps_thermal is not None: config['eps_thermal'] = args.eps_thermal
    
    # Update flags based on argparse actions
    config['gravity_on'] = bool(args.gravity) if args.gravity is not None else config['gravity_on']
    config['apply_fluctuations'] = bool(args.fluctuate) if args.fluctuate is not None else config['apply_fluctuations']
    config['do_plot'] = args.do_plot # Directly use boolean from argparse action
    config['do_print'] = args.do_print # Directly use boolean from argparse action
    
    # Update gravity vector based on flag
    if not config['gravity_on']:
        config['gravity_vec'] = np.array([0.0, 0.0, 0.0])

    # Ensure output directory exists (can also be done in save_results)
    output_dir = config.get('output_dir', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Construct full output file paths
    config['output_file_gravity'] = os.path.join(output_dir, config['output_file_base'] + "_WithGravity.mat")
    config['output_file_nogravity'] = os.path.join(output_dir, config['output_file_base'] + "_NoGravity.mat")


    # Optionally print the final config being used
    if config['do_print']:
        print("--- Simulation Configuration ---")
        import json
        print(json.dumps(config, indent=4, default=lambda x: repr(x) if isinstance(x, np.ndarray) else str(x)))
        print("------------------------------")

    return config

def get_bc_application_func(config, region_fn_lookup):
    """Returns the function responsible for applying BCs based on config."""
    mesh_id = config['mesh_id']

    if mesh_id == 1:
        # Simple hold center
        half_width = config.get('bc_hold_half_width', 0.01) # Get from config or use default
        def apply_bc(bc_object, time, X0, Nnodes):
             return fun_BC_3D_hold_center(bc_object, X0, Nnodes, 
                                          half_x=half_width, half_y=half_width)
        return apply_bc
        
    elif mesh_id == 2:
        # Peanut BC
        region_fn = region_fn_lookup.get(mesh_id, None) # Get region fn if needed
        if region_fn is None:
             print("Warning: Region function not found for mesh 2 BCs. BCs might be incorrect.")
        def apply_bc(bc_object, time, X0, Nnodes):
            return fun_BC_peanut(bc_object, X0, Nnodes,
                                   x_min = config.get('bc_peanut_xmin', 0.0577), 
                                   x_max = config.get('bc_peanut_xmax', 0.0987),
                                   y_min = config.get('bc_peanut_ymin', 0.0458), 
                                   y_max = config.get('bc_peanut_ymax', 0.0538),
                                   node_region_fn = region_fn) # Pass the looked-up region_fn
        return apply_bc

    elif mesh_id == 3:
         # Peanut BC using pattern center
        region_fn = region_fn_lookup.get(mesh_id, None)
        if region_fn is None:
             print("Warning: Region function not found for mesh 3 BCs. BCs might be incorrect.")
             
        # Need pattern center - assume it's set correctly in config during define_config
        pattern_center = config.get('Pattern_center', None) 
        if pattern_center is None:
             # Calculate if not in config (should have been done in setup ideally)
             print("Warning: Pattern_center not found in config for Mesh 3 BCs. Calculating.")
             # This requires X0 which isn't available here easily. Best to ensure setup puts it in config.
             # Placeholder: Use default from config.py if calculation is complex here.
             pattern_center = get_default_config()['pattern_params'][3]['Pattern_center']

        temp = config.get('bc_center_hold_size', 0.01)

        def apply_bc(bc_object, time, X0, Nnodes):
            return fun_BC_peanut(bc_object, X0, Nnodes,
                                  x_min = pattern_center[0]-temp, 
                                  x_max = pattern_center[0]+temp,
                                  y_min = pattern_center[1]-temp, 
                                  y_max = pattern_center[1]+temp,
                                  node_region_fn = region_fn)
        return apply_bc
    # Add elif for other mesh IDs and their specific BC logic...
    else:
        # Default: Hold the absolute center node if no specific BC defined
        print(f"Warning: No specific BC function defined for mesh {mesh_id}. Holding center node.")
        def apply_bc_default(bc_object, time, X0, Nnodes):
             x = X0[0::3]
             y = X0[1::3]
             x_mid = 0.5*(x.min() + x.max())
             y_mid = 0.5*(y.min() + y.max())
             # Find node closest to center
             dist_sq = (x - x_mid)**2 + (y - y_mid)**2
             center_node_id = np.argmin(dist_sq)
             bc_object.setBoundaryConditionNode(center_node_id, np.array([0.0, 0.0, 0.0]))
             return np.array([center_node_id]) # Return the single fixed node
        return apply_bc_default


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run thermoshell simulation.")
    parser.add_argument('--mesh', type=int, help="Which mesh ID to load (defined in config)")
    parser.add_argument('--eps-thermal', type=float, help="Target thermal strain")
    parser.add_argument('--gravity', type=int, choices=[0, 1], help="0=no gravity, 1=gravity")
    parser.add_argument('--fluctuate', type=int, choices=[0, 1], help="0=no fluctuation, 1=yes")
    parser.add_argument('--plot', dest='do_plot', action='store_true', help="Show plots during/after simulation")
    parser.add_argument('--print', dest='do_print', action='store_true', help="Enable verbose printing")
    # Add more arguments as needed (e.g., output directory, specific parameters)
    args = parser.parse_args()

    start_time = time.perf_counter()

    # --- Configuration ---
    # Defines config based on defaults and command-line arguments
    config = define_config(args)

    # --- Setup ---
    # Initializes models, loads mesh, calculates properties based on config
    # Returns a dictionary of simulation objects and data
    try:
        sim_objects = setup_simulation(config) 
    except ValueError as e:
        print(f"Error during setup: {e}")
        return # Exit if setup fails (e.g., invalid mesh ID)
    except FileNotFoundError as e:
        print(f"Error during setup: Could not find mesh file. {e}")
        return
        
    # Unpack necessary objects from the setup dictionary
    stepper = sim_objects['stepper']
    elastic_model = sim_objects['elastic_model']
    bc_object = sim_objects['bc_object']
    initial_state = sim_objects['initial_state'] 
    mesh_data = sim_objects['mesh_data'] 
    hinge_data = sim_objects['hinge_data'] 
    target_eps_th_vector = sim_objects['target_eps_th_vector']
    region_fn_lookup = sim_objects['region_fn_lookup'] # Needed for BC function selection
    save_data_base = sim_objects['save_data'] # Base data for saving output

    # --- Define BC Application Logic ---
    # Gets the correct function to apply BCs based on the mesh ID in config
    bc_apply_func = get_bc_application_func(config, region_fn_lookup)

    # --- Run Actuation Phase ---
    # Executes the main quasi-static loop, ramping thermal strain
    actuation_results = run_quasi_static_actuation(
        stepper=stepper,
        elastic_model=elastic_model,
        bc_apply_func=bc_apply_func,
        sim_params={ 
            'totalTime': config['totalTime'],
            'dt_min': config['dt_min'],
            'dt_max': config['dt_max'],
            'dt_init': config['dt_init'],
            'n_record': 2 # Record start and end
        },
        initial_state=initial_state,
        mesh_data=mesh_data, 
        hinge_data=hinge_data,
        target_eps_th_vector=target_eps_th_vector # Pass the target strain vector
    )

    # Check if actuation was successful before proceeding
    if not actuation_results.get("success", False):
        print("Actuation phase failed. Exiting.")
        # Optionally save the partial results if needed
        # save_results(config['output_file_gravity'], {**save_data_base, **actuation_results})
        return

    # --- Plotting (Actuation Result) ---
    if config['do_plot']:
        plot_truss_3d(
            actuation_results['q_final'], 
            mesh_data['ConnectivityMatrix_line'],
            NP_total=mesh_data['NP_total'],
            title=f"After Actuation (t={actuation_results.get('time_log', [0, config['totalTime']])[-1]:.4f})",
            show_labels=False
        )
        plt.show() # Control showing plot from the script

    # --- Save Actuation Results ---
    # Combine base save data with actuation results
    save_data_actuation = {**save_data_base, **actuation_results}
    save_results(config['output_file_gravity'], save_data_actuation)


    # --- Run Relaxation Phase (Optional) ---
    relaxation_results = {} # Initialize empty dict
    if config['run_relaxation']:
        # Update current state based on actuation results
        current_state_for_relax = { 
             'q_old': actuation_results['q_final'],
             'u_old': np.zeros_like(actuation_results['q_final']), # Assume static start
             'a_old': np.zeros_like(actuation_results['q_final']),
             'X0': initial_state['X0'] # Pass original coords
        }
        
        # Execute the relaxation loop, ramping down gravity
        relaxation_results = run_relaxation(
            stepper=stepper,
            elastic_model=elastic_model,
            bc_apply_func=bc_apply_func, # Use same BCs during relaxation
            relax_params={
                'relax_total_time': config['relax_total_time'],
                'relax_steps': config['relax_steps'],
                'g_initial': config['gravity_vec'] if config['gravity_on'] else np.zeros(3) # Pass initial gravity used
            },
            current_state=current_state_for_relax,
            mesh_data=mesh_data,
            hinge_data=hinge_data,
            # If appending to history, need to pass actuation history and offsets
            # For saving only final state, n_record_offset is not crucial here.
        )

        # Check for relaxation success
        if not relaxation_results.get("success", False):
             print("Relaxation phase failed.")
             # Decide whether to save partial results or exit
        else:
             # --- Plotting (Relaxation Result) ---
             if config['do_plot']:
                 plot_truss_3d(
                     relaxation_results['q_final_relax'],
                     mesh_data['ConnectivityMatrix_line'],
                     NP_total=mesh_data['NP_total'],
                     title=f"After Relaxation (Final State)",
                     show_labels=False
                 )
                 plt.show()

             # --- Save Relaxation Results ---
             # Combine base, actuation (maybe just final state), and relaxation results
             # Be careful not to save huge arrays twice if not needed
             save_data_relax = {
                 **save_data_base, 
                 'q_after_actuation': actuation_results['q_final'], # Save intermediate state
                 **relaxation_results # Save final state and logs from relaxation
                 # Decide if you want to store Q_history etc. from both phases
             }
             save_results(config['output_file_nogravity'], save_data_relax)

    # --- Final Timing ---
    end_time = time.perf_counter()
    print(f"\nTotal elapsed time: {end_time - start_time:.4f} s")

# Standard Python entry point
if __name__ == "__main__":
    main()