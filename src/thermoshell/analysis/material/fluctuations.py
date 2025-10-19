import numpy as np

def add_boundary_fluctuations(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    epsilon_th: np.ndarray,
    amp: float = 1e-3,
    n_waves: int = 4,
    decay_width: float = 0.05
) -> np.ndarray:
    """
    Add a small fluctuation to epsilon_th that is largest at the four
    edges of the rectangular domain and decays into the interior.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal coordinates
    connectivity : (Nedges,3) array
      edge list [eid, n0, n1]
    epsilon_th : (Nedges,) array
      existing thermal strains
    amp : float
      maximum fluctuation amplitude at the boundary
    n_waves : int
      how many full sin‐cycles along each edge
    decay_width : float
      distance over which the fluctuation decays to zero inwards
    """
    # 1) get midpoints back
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) domain extents
    x = mids[:,0]
    y = mids[:,1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    Lx = xmax - xmin
    Ly = ymax - ymin

    # 3) distance to each of the four edges
    d_left   = x - xmin
    d_right  = xmax - x
    d_bottom = y - ymin
    d_top    = ymax - y

    # 4) choose the closest boundary for each midpoint
    d_edge = np.minimum.reduce([d_left, d_right, d_bottom, d_top])

    # 5) build a decaying envelope exp(−d_edge/decay_width)
    envelope = np.exp(-d_edge / decay_width)

    # 6) build a sinusoidal fluctuation along the boundary coordinate:
    #    we’ll project each point onto its closest edge, then
    #    parameterize that edge by a coordinate s in [0,1].
    #    For simplicity we’ll use x/Lx for top/bottom and y/Ly for left/right.
    #    (This mixes a bit, but gives four “bands” of waves.)
    s = np.zeros_like(x)
    # where the closest is left or right, use y
    mask_v = (d_edge == d_left) | (d_edge == d_right)
    s[mask_v] = (y[mask_v] - ymin) / Ly
    # where the closest is top or bottom, use x
    mask_h = ~mask_v
    s[mask_h] = (x[mask_h] - xmin) / Lx

    # 7) fluctuation = amp * envelope * sin(2π * n_waves * s)
    fluct = amp * envelope * np.sin(2*np.pi*n_waves*s)

    # 8) return the superposition
    return epsilon_th + fluct

def add_OneSide_boundary_fluctuations(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    epsilon_th: np.ndarray,
    amp: float = 1e-3,
    n_waves: int = 4,
    decay_width: float = 0.05
) -> np.ndarray:
    """
    Add a small fluctuation to epsilon_th that is largest at the four
    edges of the rectangular domain and decays into the interior.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal coordinates
    connectivity : (Nedges,3) array
      edge list [eid, n0, n1]
    epsilon_th : (Nedges,) array
      existing thermal strains
    amp : float
      maximum fluctuation amplitude at the boundary
    n_waves : int
      how many full sin‐cycles along each edge
    decay_width : float
      distance over which the fluctuation decays to zero inwards
    """
    # 1) get midpoints back
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) domain extents
    x = mids[:,0]
    y = mids[:,1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    Lx = xmax - xmin
    Ly = ymax - ymin

    # 3) distance to each of the four edges
    d_left   = x - xmin
    d_right  = xmax - x
    d_bottom = y - ymin
    d_top    = ymax - y

    # 4) choose the closest boundary for each midpoint
    d_edge = np.minimum.reduce([d_left])

    # 5) build a decaying envelope exp(−d_edge/decay_width)
    envelope = np.exp(-d_edge / decay_width)

    # 6) build a sinusoidal fluctuation along the boundary coordinate:
    #    we’ll project each point onto its closest edge, then
    #    parameterize that edge by a coordinate s in [0,1].
    #    For simplicity we’ll use x/Lx for top/bottom and y/Ly for left/right.
    #    (This mixes a bit, but gives four “bands” of waves.)
    s = np.zeros_like(x)
    # where the closest is left or right, use y
    mask_v = (d_edge == d_left) | (d_edge == d_right)
    s[mask_v] = (y[mask_v] - ymin) / Ly
    # where the closest is top or bottom, use x
    mask_h = ~mask_v
    s[mask_h] = (x[mask_h] - xmin) / Lx

    # 7) fluctuation = amp * envelope * sin(2π * n_waves * s)
    fluct = amp * envelope * np.sin(2*np.pi*n_waves*s)

    # 8) return the superposition
    return epsilon_th + fluct