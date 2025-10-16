import numpy as np

def fun_edge_lengths(nodeXYZ: np.ndarray,
                         Connectivity: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean length of each edge.

    Parameters
    ----------
    nodeXYZ : array, shape (Nnodes, 4)
        Columns are [nodeID, x, y, z].
    Connectivity : array, shape (Nedges, 3)
        Columns are [edgeID, node_i, node_j].

    Returns
    -------
    edge_lengths : array, shape (Nedges,)
        edge_lengths[k] is the length of the edge whose ID is k.
    """
    coords = nodeXYZ[:, 1:4]     # shape (Nnodes, 3)

    Nedges = Connectivity.shape[0]
    edge_lengths = np.zeros(Nedges)

    for eid, ni, nj in Connectivity.astype(int):
        p_i = coords[ni,:]
        p_j = coords[nj,:]
        edge_lengths[eid] = np.linalg.norm(p_j - p_i)

    return edge_lengths