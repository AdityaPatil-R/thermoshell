import numpy as np
from typing import Callable, Tuple

def assign_thermal_strains_contour(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_thermal: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    inside: bool = True
) -> np.ndarray:
    """
    Like assign_thermal_strains, but replaces the flat step
    by a smooth ramp that goes from 0 at the region boundary
    up to eps_thermal at the point farthest from that boundary
    (and vice-versa if inside=False).

    Parameters
    ----------
    node_coords : (Nnodes,3) array
        nodal (x,y,z)
    connectivity : (Nedges,3) array
        each row [eid,n0,n1]
    eps_thermal : float
        peak thermal strain
    region_fn : callable
        given mids=(Nedges,3) returns a boolean mask
    inside : bool
        if True, ramp *inside* the mask; else ramp *outside*

    Returns
    -------
    epsilon_th : (Nedges,) array
    """
    # 1) midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) build mask
    mask = region_fn(mids)

    # 3) select the region we want to ramp over
    if inside:
        ramp_idx    = np.nonzero(mask)[0]
        boundary_idx = np.nonzero(~mask)[0]
    else:
        ramp_idx    = np.nonzero(~mask)[0]
        boundary_idx = np.nonzero(mask)[0]

    # 4) if either set is empty, fall back to flat
    if ramp_idx.size == 0 or boundary_idx.size == 0:
        flat = eps_thermal if inside else 0.0
        out  = np.full(mids.shape[0], flat, dtype=float)
        return out

    # 5) for each point in the ramp region, find its distance
    #    to the nearest point of the *other* region → d_i
    P_ramp     = mids[ramp_idx]     # (R,3)
    P_boundary = mids[boundary_idx] # (B,3)

    # compute pairwise squared‐distances R×B
    #    (this can handle a few thousand edges in a few seconds)
    diffs = P_ramp[:, None, :] - P_boundary[None, :, :]
    d2    = np.einsum('rbi,rbi->rb', diffs, diffs)
    d_min = np.sqrt(np.min(d2, axis=1))  # (R,)

    # 6) normalize so that the farthest ramp‐point has weight=1
    d_max = d_min.max()
    if d_max <= 0:
        # degenerate → flat
        weights = np.ones_like(d_min)
    else:
        weights = d_min / d_max

    # 7) build output
    epsilon_th = np.zeros(mids.shape[0], dtype=float)
    if inside:
        # inside the mask: 0 at the boundary, eps_thermal at farthest interior
        epsilon_th[ramp_idx] = eps_thermal * (1.0 - weights)
    else:
        # outside the mask: eps_thermal at the boundary, 0 at farthest outside
        epsilon_th[ramp_idx] = eps_thermal * weights

    return epsilon_th


def assign_thermal_strains_LinearTraisition(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_min: float,
    eps_max: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    inside: bool = True
) -> np.ndarray:
    """
    Assign thermal strain to each edge whose midpoint lies inside (or outside)
    the region defined by `region_fn`, but instead of a flat eps_thermal we
    do a linear ramp from eps_max at the mesh mid-height line to eps_min at
    the top/bottom edges.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal (x,y,z)
    connectivity : (Nedges,3) array
      each row [eid, n0, n1]
    eps_min : float
      thermal strain at the top/bottom of the mesh (t=1)
    eps_max : float
      thermal strain on the mid-height line (t=0)
    region_fn : callable
      given mids=(Nedges,3) array returns boolean mask
    inside : bool
      if True, apply ramp inside mask; else apply ramp outside
    """
    # 1) midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # shape (Nedges,3)

    # 2) region mask
    mask = region_fn(mids)

    # 3) compute normalized vertical dist t in [0,1]
    y_all  = node_coords[:,1]
    y_min  = y_all.min()
    y_max  = y_all.max()
    y_mid  = 0.5*(y_min + y_max)
    half_h = 0.5*(y_max - y_min)

    dy     = np.abs(mids[:,1] - y_mid)
    t_norm = np.clip(dy/half_h, 0.0, 1.0)

    # 4) allocate and fill
    epsilon_th = np.zeros(connectivity.shape[0], dtype=float)

    # interpolation formula: at t=0 → eps_min, at t=1 → eps_max
    ramp = eps_min + (eps_max - eps_min) * t_norm

    if inside:
        epsilon_th[mask]  = ramp[mask]
    else:
        epsilon_th[~mask] = ramp[~mask]

    return epsilon_th


def assign_thermal_strains_RadialTransition(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_min: float,
    eps_max: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    inside: bool = True
) -> np.ndarray:
    """
    Assign thermal strain to each edge whose midpoint lies inside (or outside)
    the region defined by `region_fn`, with a radial linear ramp from
    eps_max at the mesh center down to eps_min at the region boundary.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal (x,y,z)
    connectivity : (Nedges,3) array
      each row [eid, n0, n1]
    eps_min : float
      thermal strain at the farthest mask boundary (r = r_max)
    eps_max : float
      thermal strain at the mesh center (r = 0)
    region_fn : callable
      given mids=(Nedges,3) returns boolean mask
    inside : bool
      if True, apply ramp inside mask; else apply ramp outside
    """
    # 1) compute midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) build mask
    mask = region_fn(mids)   # boolean (Nedges,)

    # 3) compute mesh center in x–y
    x_all = node_coords[:,0]
    y_all = node_coords[:,1]
    x_mid = 0.5*(x_all.min() + x_all.max())
    y_mid = 0.5*(y_all.min() + y_all.max())

    # 4) compute radial distances for each midpoint
    dx    = mids[:,0] - x_mid
    dy    = mids[:,1] - y_mid
    r_all = np.sqrt(dx*dx + dy*dy)

    # 5) find maximum radius *within* the mask (so we ramp only to its boundary)
    if inside:
        r_max = r_all[mask].max() if np.any(mask) else 0.0
    else:
        r_max = r_all[~mask].max() if np.any(~mask) else 0.0

    # avoid division by zero
    if r_max <= 0:
        # all r==0 or empty region → flat eps_max or eps_min
        flat_value = eps_max if inside else eps_min
        return np.full(connectivity.shape[0], flat_value, dtype=float)

    # 6) normalized radius in [0,1]
    t_norm = np.clip(r_all / r_max, 0.0, 1.0)

    # 7) build ramp: at r=0 → eps_max; at r=r_max → eps_min
    ramp = eps_min + (eps_max - eps_min) * t_norm

    # 8) fill output
    epsilon_th = np.zeros(connectivity.shape[0], dtype=float)
    if inside:
        epsilon_th[mask]  = ramp[mask]
    else:
        epsilon_th[~mask] = ramp[~mask]

    return epsilon_th


def assign_thermal_strains_EllipticTransition(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_min: float,
    eps_max: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    ellipse_axes: Tuple[float,float] = (1.0,1.0),
    inside: bool = True
) -> np.ndarray:
    """
    Same as your radial ramp, but distance is measured in an ellipse:
      ((x - x_mid)/a)^2 + ((y - y_mid)/b)^2 = 1

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal (x,y,z)
    connectivity : (Nedges,3) array
      each row [eid, n0, n1]
    eps_min : float
      thermal strain at the ellipse boundary (t_norm = 1)
    eps_max : float
      thermal strain at the center      (t_norm = 0)
    region_fn : callable
      given mids=(Nedges,3) returns boolean mask
    ellipse_axes : (a,b)
      semi-axes of the ellipse in x- and y-directions
    inside : bool
      if True, apply ramp inside mask; else apply ramp outside
    """
    # 1) edge midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) boolean mask from user
    mask = region_fn(mids)   # (Nedges,)

    # 3) compute global mesh center
    x_all = node_coords[:,0]
    y_all = node_coords[:,1]
    x_mid = 0.5*(x_all.min() + x_all.max())
    y_mid = 0.5*(y_all.min() + y_all.max())

    # 4) compute normalized elliptical radius
    a, b = ellipse_axes
    dx   = mids[:,0] - x_mid
    dy   = mids[:,1] - y_mid
    # ellipse “radius” r_ell in [0..∞):
    r_ell = np.sqrt((dx/a)**2 + (dy/b)**2)

    # 5) pick the max ellipse‐radius *within* the mask (or its complement)
    if inside:
        r_max = r_ell[mask].max()  if np.any(mask)  else 0.0
    else:
        r_max = r_ell[~mask].max() if np.any(~mask) else 0.0

    # 6) handle degenerate case
    if r_max <= 0:
        flat = eps_max if inside else eps_min
        return np.full(connectivity.shape[0], flat, dtype=float)

    # 7) normalize to [0..1], build linear ramp eps=eps_max→eps_min
    t_norm = np.clip(r_ell / r_max, 0.0, 1.0)
    ramp   = eps_min + (eps_max - eps_min)*t_norm

    # 8) fill only the mask (or its complement)
    epsilon_th = np.zeros(connectivity.shape[0], dtype=float)
    if inside:
        epsilon_th[mask]   = ramp[mask]
    else:
        epsilon_th[~mask]  = ramp[~mask]

    return epsilon_th


def assign_youngs_modulus_v3(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    region_fn: Callable[[np.ndarray], np.ndarray],
    circle_center: Tuple[float,float],
    circle_radius: float,
    Ysoft: float,
    Yhard: float,
    Yratio: float,
    inside: bool = True,
    *,
    x_thresh_left: float = None,
    x_thresh_right: float = None,
    hard_factor: float = 1.0
) -> np.ndarray:
    """
    Build a (Nedges,) array of Young's moduli with two effects:
      1) a soft/hard radial ramp about circle_center (as before),
      2) anywhere x > x_thresh, force Y = hard_factor * Yhard_r.

    Parameters
    ----------
    node_coords : (Nnodes,3)
    connectivity: (Nedges,3)
    region_fn   : mids -> boolean (mask)
    circle_center: (cx,cy)
    circle_radius: R
    Ysoft, Yhard, Yratio: ramp endpoints
    inside      : apply inside-mask ramp if True, else complement
    x_thresh    : if not None, x > x_thresh ⇒ override
    hard_factor : factor to multiply Yhard_r by in override zone
    """
    # 1) compute midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)
    x    = mids[:,0]

    # 2) mask & midpoint‐radius for ramp
    mask  = region_fn(mids)   # (Nedges,)
    cx, cy = circle_center
    dx = mids[:,0] - cx
    dy = mids[:,1] - cy
    r  = np.sqrt(dx*dx + dy*dy)
    r_norm = np.clip(r / circle_radius, 0.0, 1.0)

    # 3) soft/hard endpoints
    Ysoft_r = Ysoft
    Ysoft_R = Ysoft * Yratio
    Yhard_r = Yhard
    Yhard_R = Yhard * Yratio

    # 4) build the ramp arrays
    soft_ramp = Ysoft_r + (Ysoft_R - Ysoft_r) * r_norm
    hard_ramp = Yhard_r + (Yhard_R - Yhard_r) * r_norm

    # 5) allocate and fill
    Y = np.empty_like(r)
    if inside:
        Y[mask]  = soft_ramp[mask]
        Y[~mask] = hard_ramp[~mask]
    else:
        Y[mask]  = hard_ramp[mask]
        Y[~mask] = soft_ramp[~mask]

    # 6) override to super‐hard on the right side
    if x_thresh_left is not None:
        
        in_band = (x > x_thresh_left) & (x < x_thresh_right)
        if inside:
            override = (~mask) & in_band
        else:
            override = (mask) & in_band

        Y[override] = hard_factor * Yhard_r

    return Y