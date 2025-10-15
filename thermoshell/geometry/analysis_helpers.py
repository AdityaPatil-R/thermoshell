import numpy as np
from typing import Tuple, List

# --- 1. Edge Length Calculation ---

def fun_edge_lengths(nodeXYZ: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean length of each edge in the reference configuration.

    Parameters:
        nodeXYZ (np.ndarray): Columns are [nodeID, x, y, z] (Nnodes, 4).
        connectivity (np.ndarray): Columns are [edgeID, node_i, node_j] (Nedges, 3).

    Returns:
        np.ndarray: edge_lengths (Nedges,), where index k is the length of 
                    the edge whose ID is k.
    """
    # Use 0-based indexing for coordinates (columns 1:4)
    coords = nodeXYZ[:, 1:4]  # shape (Nnodes, 3)

    Nedges = connectivity.shape[0]
    edge_lengths = np.zeros(Nedges)

    for eid, ni, nj in connectivity.astype(int):
        # ni and nj are the 0-based node indices
        p_i = coords[ni, :]
        p_j = coords[nj, :]
        edge_lengths[eid] = np.linalg.norm(p_j - p_i)

    return edge_lengths


# --- 2. Post-Processing Analysis (Tip Deflection and Reaction) ---

def fun_reaction_force_RightEnd(X0_4columns: np.ndarray, R_history: np.ndarray,
                                step: int = -1, axis: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the total reaction force along a specific axis at the far right 
    edge (maximum coordinate along the X-axis) of the mesh.

    Parameters:
        X0_4columns (np.ndarray): Reference nodal coordinates [ID, X, Y, Z].
        R_history (np.ndarray): Reaction force history (Nsteps, Ndofs).
        step (int): The history step index to analyze (default: -1, final step).
        axis (int): The component of force to sum (0=X, 1=Y, 2=Z).

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: 
            (reaction_sum, right_nodes, dof_indices)
    """
    # We find the 'right end' based on the X-coordinate (index 1 in X0_4columns)
    x_coords = X0_4columns[:, 1]
    xmax = x_coords.max()
    
    # Node indices located at the maximum X-coordinate
    right_nodes = np.where(np.isclose(x_coords, xmax, atol=1e-8))[0]
    
    # Calculate global DOF index for the desired axis (3*node_idx + axis)
    dof_indices = right_nodes * 3 + axis
    
    # Sum the reactions for the nodes and axis at the specified step
    reaction_sum = np.sum(R_history[step, dof_indices])
    
    return reaction_sum, right_nodes, dof_indices


def fun_deflection_RightEnd(X0_4columns: np.ndarray, Q_history: np.ndarray,
                            step: int = -1, axis: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the nodal deflections (displacement from reference) along a specific axis 
    at the far right end (maximum coordinate along the X-axis) of the mesh.

    Parameters:
        X0_4columns (np.ndarray): Reference nodal coordinates [ID, X, Y, Z].
        Q_history (np.ndarray): Final nodal coordinates history (Nsteps, Ndofs).
        step (int): The history step index to analyze (default: -1, final step).
        axis (int): The component of displacement (0=X, 1=Y, 2=Z).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (deflections, right_nodes, dof_indices)
    """
    # Find the nodes at the maximum X-coordinate (index 1 in X0_4columns)
    x_coords = X0_4columns[:, 1]
    xmax = x_coords.max()
    
    right_nodes = np.where(np.isclose(x_coords, xmax, atol=1e-8))[0]
    
    # Calculate global DOF index for the desired axis
    dof_indices = right_nodes * 3 + axis
    
    q_step = Q_history[step]
    
    # Reference coordinates (X, Y, Z are columns 1, 2, 3 in X0_4columns)
    X0_flat = X0_4columns[:, 1:4].ravel()
    
    # Deflection = Current Position - Reference Position
    deflections = q_step[dof_indices] - X0_flat[dof_indices]
    
    return deflections, right_nodes, dof_indices
