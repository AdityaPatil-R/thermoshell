# src/thermoshell/config.py

import numpy as np

def get_default_config():
    """Returns a dictionary containing the default simulation parameters."""
    
    config = {}

    # --- Mesh / Geometry ---
    config['mesh_id'] = 1 # Default mesh ID
    config['mesh_files'] = {
        1: 'data/mesh_python_circle_970nodes_scale100mm.txt',
        2: 'data/mesh_rectangle_scaled_1215nodes_scale100mm_155mm.txt',
        3: 'data/mesh_python_square_388nodes_scale100mm.txt',
        # Add paths for other meshes (4-9) if they exist
    }
    # Default path (will be overwritten based on mesh_id later)
    config['mesh_file'] = config['mesh_files'][config['mesh_id']] 

    # --- Material Properties ---
    config['Ysoft'] = 1.0e6
    config['Yhard'] = 3.0e6
    config['FactorKs'] = 10.0
    config['FactorKb'] = 1.0
    
    # Bilayer params (lk, h1, h2) specific to mesh
    config['bilayer_params'] = { 
        1: {'lk': 0.0032, 'h1': 0.3e-3, 'h2': 0.7e-3}, 
        2: {'lk': 0.0040, 'h1': 0.3e-3, 'h2': 0.6e-3},
        3: {'lk': 0.0058, 'h1': 0.3e-3, 'h2': 1.0e-3},
        # Add params for other meshes if needed
    }
    # Set default lk, h1, h2 based on default mesh_id
    config.update(config['bilayer_params'][config['mesh_id']])

    # --- Patterning ---
    config['eps_thermal'] = -0.3 # Default target thermal strain
    config['apply_fluctuations'] = False # Default fluctuation flag
    
    # Mesh-specific patterning parameters
    config['pattern_params'] = {
        1: { # Circle Mesh
            'Yratio': 1.0,   
            'OuterR': 0.05, 
            'Pattern_center': [0.0, 0.0], # Can be calculated later based on mesh bounds
            'StripeWidth': 0.0059,   
            'StripeLength': 0.0441, 
            'Stripe_r': 0.0029,
            'pattern_type': 'circle_six_arms' # Identifier for which region_fn to use
        },
        2: { # Rectangle Mesh
            'Yratio': 1.0,
            'OuterR': 0.05,
            'Pattern_center': None, # To be calculated based on mesh bounds
            'delta_shape': 0.02787,
            'n_spokes': 6,  
            'star_radius': 0.044595,
            'star_thickness': 0.002787,
            'beam_thickness': 0.002787,
            'Y_override_params': { # Parameters for assign_youngs_modulus_v3
                 'x_thresh_left': 0.10,
                 'x_thresh_right': 0.185,
                 'hard_factor': 1.0,
             },
            'pattern_type': 'whole_peanut'
        },
        3: { # Square Mesh
             'Yratio': 1.0,
             'OuterR': 0.05,
             'Pattern_center': [0.05, 0.05], # Specific center for this mesh
             'StripeWidth': 0.0059,
             'StripeLength': 0.0707,
             'Stripe_r': 0.0006,
             'pattern_type': 'square_X'
        },
         # Add params for other meshes if needed
    }
    # Add default pattern params to top level config
    config.update(config['pattern_params'][config['mesh_id']])


    # --- Solver / Simulation Control ---
    config['gravity_on'] = True
    config['gravity_vec'] = np.array([0.0, 0.0, 9.81]) # Physical direction (sign flip handled in solver/setup if needed)
    config['nodal_mass'] = 1e-7 
    config['qtol'] = 1e-5
    config['maxIter'] = 20
    config['totalTime'] = 1.0 # Duration for actuation phase
    config['dt_init'] = 0.05
    config['dt_min'] = 1e-8
    config['dt_max'] = 0.1
    config['beta'] = 1.0 # Stretch-bending coupling factor
    config['theta_bar'] = 0.0 # Target dihedral angle (usually zero for flat initial state)

    # --- Relaxation Phase ---
    config['run_relaxation'] = True 
    config['relax_total_time'] = 1.0
    config['relax_steps'] = 10
    
    # --- Output ---
    # Define default output directory and base filename (can be formatted later)
    config['output_dir'] = 'output' 
    config['output_file_base'] = f"output_mesh{config['mesh_id']}" 
    
    # --- Plotting/Verbosity (from original script) ---
    config['do_print'] = False # Corresponds to iPrint
    config['do_plot'] = False  # Corresponds to iPlot
    config['run_tests'] = False # Corresponds to iTest (tests should be separate)

    return config

# --- You could add functions here to load/save config from/to files (e.g., YAML) ---
# def load_config_from_yaml(filepath):
#     import yaml
#     with open(filepath, 'r') as f:
#         user_config = yaml.safe_load(f)
#     
#     # Merge user_config with defaults
#     config = get_default_config()
#     config.update(user_config) # Simple merge, could be deeper
#     # Re-select mesh-specific params based on potentially updated mesh_id
#     mesh_id = config['mesh_id']
#     config['mesh_file'] = config['mesh_files'].get(mesh_id, None)
#     if config['mesh_file'] is None: raise ValueError(...)
#     config.update(config['bilayer_params'].get(mesh_id, {}))
#     config.update(config['pattern_params'].get(mesh_id, {}))
#     
#     return config

if __name__ == '__main__':
    # Example of how to get the default config
    default_config = get_default_config()
    print("Default Configuration:")
    import json
    print(json.dumps(default_config, indent=4, default=lambda x: repr(x) if isinstance(x, np.ndarray) else x))

    # Example: Override mesh_id and see changes
    test_config = default_config.copy()
    test_config['mesh_id'] = 2
    # --- Logic to update dependent defaults based on mesh_id ---
    # This logic would live in your main script or setup function typically
    mesh_id = test_config['mesh_id']
    test_config['mesh_file'] = test_config['mesh_files'].get(mesh_id, "Invalid Mesh ID")
    test_config.update(test_config['bilayer_params'].get(mesh_id, {})) # Update lk, h1, h2
    test_config.update(test_config['pattern_params'].get(mesh_id, {})) # Update pattern specific params
    test_config['output_file_base'] = f"output_mesh{mesh_id}"
    # --- End update logic ---
    
    print("\nConfiguration after setting mesh_id = 2:")
    print(json.dumps(test_config, indent=4, default=lambda x: repr(x) if isinstance(x, np.ndarray) else x))