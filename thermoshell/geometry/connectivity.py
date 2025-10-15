import numpy as np
from collections import defaultdict, Counter
from typing import Tuple, List, Dict

# --- 1. Edge-to-ID Lookup Helper ---

def get_edge_map(connectivity: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    Creates a dictionary mapping an undirected node pair (n_min, n_max) to its 
    corresponding Edge ID (eid). Used to quickly find the edge ID and related 
    properties (like thermal strain) for any pair of nodes.

    Parameters:
        connectivity (np.ndarray): Array of shape (Nedges, 3) with [edgeID, n0, n1].

    Returns:
        Dict[Tuple[int, int], int]: Mapping from sorted node tuple to edge ID.
    """
    edge_dict = {}
    for eid, n0, n1 in connectivity.astype(int):
        # Store the node pair sorted as the key (undirected edge)
        edge_dict[tuple(sorted((n0, n1)))] = eid
    return edge_dict

# --- 2. Hinge Connectivity Derivation ---

def get_hinge_connectivity(
    node_xyz: np.ndarray,
    connectivity: np.ndarray,
    triangles: np.ndarray
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Identifies interior edges (hinges) and determines the ordered four 
    nodes [n0, n1, oppA, oppB] required for dihedral angle calculations.

    Parameters:
        node_xyz (np.ndarray): (Nnodes, 4) array: [ID, x, y, z].
        connectivity (np.ndarray): (Nedges, 3) array: [edgeID, n0, n1].
        triangles (np.ndarray): (Ntris, 4) array: [triID, n1, n2, n3].

    Returns:
        Tuple[np.ndarray, Dict]:
            HingeQuads_order (Nhinges, 5): [eid, n0, n1, oppA, oppB] ordered nodes.
            edge_dict (Dict): Map of (n_min, n_max) -> eid.
    """
    # Use 0-based indices for all internal calculations
    tri_indices = triangles[:, 1:4].astype(int)
    conn_indices = connectivity[:, 1:3].astype(int)
    
    # 1. Find all interior edges (hinges)
    tri_edges = []
    edge_to_opps = defaultdict(list)
    
    for tri in tri_indices:
        # Get edges of the triangle
        v1, v2, v3 = tri
        edges = [tuple(sorted((v1, v2))), tuple(sorted((v2, v3))), tuple(sorted((v3, v1)))]
        
        for a, b in edges:
            # The opposite node is the one not in the edge (a, b)
            opp = next(v for v in tri if v not in (a, b))
            edge_to_opps[a, b].append(opp)
            tri_edges.append(a, b)
    
    # Edges appearing exactly twice are interior hinges
    edge_counts = Counter(tri_edges)
    hinge_keys = {edge for edge, cnt in edge_counts.items() if cnt == 2}
    
    # 2. Get the Edge IDs (EIDs) corresponding to these hinges
    edge_dict = get_edge_map(connectivity)
    
    hinge_eids = []
    for eid, n0, n1 in connectivity.astype(int):
        if tuple(sorted((n0, n1))) in hinge_keys:
            hinge_eids.append(eid)
    
    # --- 3. Build the initial HingeQuads list [eid, n0, n1, oppA, oppB] ---
    hinge_quads = []
    
    # Dictionary from node pair to two opposite nodes
    # We must iterate over original connectivity to preserve the eid
    for eid in hinge_eids:
        n0, n1 = conn_indices[eid]
        key = tuple(sorted((n0, n1)))
        
        # The two opposite nodes for this hinge
        oppA, oppB = edge_to_opps[key]
        hinge_quads.append([eid, n0, n1, oppA, oppB])

    # --- 4. Order the HingeQuads for consistent dihedral angle sign ---
    # The convention: normals n0 and n1 should both point generally in 
    # the positive Z direction (or both negative Z, but consistently).
    
    HingeQuads_order = []
    # Use coordinates from column 1:4 (x, y, z)
    coords = node_xyz[:, 1:4]
    
    for eid, n0, n1, oppA, oppB in hinge_quads:
        x0 = coords[n0, :]
        x1 = coords[n1, :]
        x2 = coords[oppA, :]
        x3 = coords[oppB, :]

        # Vectors for Triangle 0: (x0, x1, x2)
        m_e0_0 = x1 - x0
        m_e1_0 = x2 - x0
        # Normal n0 = (x1-x0) x (x2-x0) = m_e0 x m_e1
        n0_v = np.cross(m_e0_0, m_e1_0)
        
        # Vectors for Triangle 1: (x0, x1, x3)
        m_e0_1 = x1 - x0 # Same hinge vector
        m_e1_1 = x3 - x0
        # Normal n1 = (x3-x0) x (x0-x1) is used for angle calc, but here we use 
        # m_e0 x m_e1 to check orientation against Z axis.
        # Normal n1 = (x0-x1) x (x3-x1) (different definition, use triangle vertices)
        
        # Consistent ordering for dihedral requires the triangles (n0, n1, x2) and (n0, n1, x3)
        # to have normals that point generally in the same half-space (e.g., both z>0).
        # Normal for triangle 1 must be (x1-x0) x (x3-x0) or similar. 
        # Using cross(v_hinge, v_side) for normal orientation check.
        
        # Normal 1: Cross product across hinge (x0 -> x1) and node x3 (oppB)
        n1_v = np.cross(x1 - x0, x3 - x0) 
        
        # Check orientation via Z-component sign
        z_sign_A = np.sign(n0_v[2])
        z_sign_B = np.sign(n1_v[2])

        # If signs are opposite, swap oppA and oppB to ensure consistency
        # If both normals point downward (< 0), swap oppA and oppB to flip their definitions 
        # to ensure the dihedral angle calculation is consistent.
        
        if z_sign_A != z_sign_B:
             # This indicates an issue with the initial mesh geometry/numbering 
             # where the triangles lie in opposite half-spaces relative to Z=0.
             # We rely on the mesh generator to give sensible initial triangles.
             # For flat meshes (z=0), this check is unreliable. We just default
             # to the initial assignment for flat meshes.
             HingeQuads_order.append([eid, n0, n1, oppA, oppB]) # Keep initial order
        
        elif z_sign_A < 0 and z_sign_B < 0:
            # Both normals point down. Swap oppA and oppB to flip n0_v and n1_v sign 
            # for a consistent positive angle calculation in dihedral_helpers.
            HingeQuads_order.append([eid, n0, n1, oppB, oppA])
            
        else: # z_sign_A >= 0 and z_sign_B >= 0 (or flat)
            # Normals point up or are flat. Keep original order.
            HingeQuads_order.append([eid, n0, n1, oppA, oppB])


    HingeQuads_order = np.array(HingeQuads_order, dtype=int)
    
    return HingeQuads_order, edge_dict
