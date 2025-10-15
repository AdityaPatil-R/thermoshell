# %% Effective ks and kb for a bilayer composite

def bilayer_flexural_rigidity(h1, h2, Y1, Y2, b1=1.0, b2=1.0):
    """
    Computes the effective flexural rigidity D_eff and the discrete hinge stiffness k_b
    for a bilayer composite.
    """
    # layer centroids (measured from bottom of layer 1)
    y1 = h1 / 2.0
    y2 = h1 + h2 / 2.0

    # neutral axis location
    num = Y1 * b1 * h1 * y1 + Y2 * b2 * h2 * y2
    den = Y1 * b1 * h1     + Y2 * b2 * h2
    ybar = num / den

    # individual layer contributions to D
    D1 = Y1 * b1 * (h1**3 / 12.0 + h1 * (y1 - ybar)**2)
    D2 = Y2 * b2 * (h2**3 / 12.0 + h2 * (y2 - ybar)**2)

    # total effective rigidity and hinge stiffness
    D_eff = D1 + D2

    return D_eff

def assign_youngs_modulus(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    region_fn: Callable[[np.ndarray], np.ndarray],
    circle_center: Tuple[float,float],
    circle_radius: float,
    Ysoft: float,
    Yhard: float,
    Yratio: float,
    inside: bool = True
) -> np.ndarray:
    """
    Build a (Nedges,) array of Young's moduli:
      Ysoft inside the region, Yhard outside (or vice versa).
    """
    # 1) compute midpoints
    n0  = connectivity[:,1].astype(int)
    n1  = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])   # shape (Nedges,3)

    # 2) test region
    mask = region_fn(mids)   # boolean (Nedges,)

    # 3) piecewise assign *with* a radial ramp inside the region
    Y = np.empty(connectivity.shape[0], dtype=float)

    # --- compute radial distances from the circle center ---
    # You must know your circle_center=(cx,cy) and circle_radius=R
    cx, cy = circle_center
    R = circle_radius

    dx = mids[:,0] - cx
    dy = mids[:,1] - cy
    r  = np.sqrt(dx*dx + dy*dy)

    # normalized radius [0..1]
    r_norm = np.clip(r / R, 0.0, 1.0)

    # define the two soft‐moduli endpoints
    Ysoft_R  = Yratio * Ysoft    # at r = R
    Ysoft_r = Ysoft           # at r = 0
    Yhard_R  = Yratio * Yhard
    Yhard_r = Yhard

    if inside:
        Y[mask]  = Ysoft_r + (Ysoft_R - Ysoft_r) * r_norm[mask]
        # outside: hard material
        Y[~mask] = Yhard_r + (Yhard_R - Yhard_r) * r_norm[~mask]
    else:
        # inside: hard material
        Y[mask]  = Yhard_r + (Yhard_R - Yhard_r) * r_norm[mask]
        Y[~mask] = Ysoft_r + (Ysoft_R - Ysoft_r) * r_norm[~mask]

    return Y


def assign_youngs_modulus_v2(
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
    x_thresh: float = None,
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
    inside      : apply inside‐mask ramp if True, else complement
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
    if x_thresh is not None:
        if inside:
            override = (~mask) & (x > x_thresh)
        else:
            override = (mask) & (x > x_thresh)

        Y[override] = hard_factor * Yhard_r

    return Y


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
    inside      : apply inside‐mask ramp if True, else complement
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