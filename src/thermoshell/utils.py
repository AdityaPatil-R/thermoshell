# src/thermoshell/utils.py

import scipy.io as sio
import os
from typing import Dict, Any

def save_results(filepath: str, results_dict: Dict[str, Any], verbose: bool = True):
    """
    Saves the results dictionary to a .mat file using scipy.io.savemat.

    Args:
        filepath (str): The full path (including filename) to save the .mat file.
        results_dict (Dict[str, Any]): The dictionary containing simulation results
                                       (NumPy arrays, lists, scalars, etc.).
        verbose (bool): If True, print confirmation message.
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if verbose:
                print(f"Created output directory: {output_dir}")

        # Save the data using scipy.io.savemat
        sio.savemat(filepath, results_dict, do_compression=True)

        if verbose:
            print(f"Results successfully saved to: {filepath}")
            # Optionally list keys saved:
            # print("  Keys saved: ", ", ".join(results_dict.keys()))

    except Exception as e:
        print(f"Error saving results to {filepath}: {e}")
        # Optionally re-raise the exception if the calling code should handle it
        # raise

# Example usage (can be removed or kept under if __name__ == '__main__'):
if __name__ == '__main__':
    import numpy as np
    print("Example usage of save_results:")

    # Create dummy data
    dummy_results = {
        'Q_history': np.random.rand(10, 5),
        'time_log': np.linspace(0, 1, 10),
        'final_state': {'q': np.random.rand(5), 'info': 'Example'},
        'config_param': 'value_example'
    }
    dummy_filepath = 'output/example_run/dummy_results.mat'

    # Save the dummy data
    save_results(dummy_filepath, dummy_results)

    # Example of trying to load it back (optional)
    try:
        loaded_data = sio.loadmat(dummy_filepath)
        print(f"\nSuccessfully loaded back {dummy_filepath}.")
        # Note: MATLAB saves Python dicts as structured arrays
        # Accessing nested dicts might require extra steps like loaded_data['final_state'][0,0]['q']
        print("  Keys loaded:", loaded_data.keys())
        print("  Shape of loaded Q_history:", loaded_data['Q_history'].shape)
    except Exception as e:
        print(f"Error loading back {dummy_filepath}: {e}")