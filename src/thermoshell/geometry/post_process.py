import numpy as np

def fun_reaction_force_RightEnd(X0_4columns, R_history, step=-1, axis=1):
    # Compute the total vertical reaction at the right end of the beam.
   
    coords = X0_4columns[:, axis]
    xmax   = coords.max()
    
    right_nodes = np.where(np.isclose(coords, xmax, atol=1e-8))[0]
    dof_z = right_nodes * 3 + 2

    reaction_sum = np.sum(R_history[step, dof_z])
    return reaction_sum, right_nodes, dof_z


def fun_EBBeam_reaction_force(delta, E, h, X0_4columns, axis_length=1, axis_width=2):
    # Compute Euler–Bernoulli reaction force for a cantilever beam under end deflection.

    # Beam length
    coords_L = X0_4columns[:, axis_length]
    L = coords_L.max() - coords_L.min()
    # Beam width
    coords_b = X0_4columns[:, axis_width]
    b = coords_b.max() - coords_b.min()
    
    I = b * h**3 / 12.0
    # Point-load formula: δ = P L^3 / (3 E I) -> P = 3 E I δ / L^3
    P = 3.0 * E * I * delta / L**3
    return P


def fun_deflection_RightEnd(X0_4columns, Q_history, step=-1, axis=1):
    # Compute the vertical deflections at the right end of the beam.

    # extract the coordinate along the beam axis
    coords = X0_4columns[:, axis]
    xmax   = coords.max()
    
    # find the nodes at the right tip
    right_nodes = np.where(np.isclose(coords, xmax, atol=1e-8))[0]
    # global z‐DOF indices for those nodes
    dof_z = right_nodes * 3 + 2
    
    q_step = Q_history[step]
    X0_flat = X0_4columns[:,1:4].ravel()
    
    # compute deflection = current_z - reference_z
    deflections = q_step[dof_z] - X0_flat[dof_z]
    
    return deflections, right_nodes, dof_z