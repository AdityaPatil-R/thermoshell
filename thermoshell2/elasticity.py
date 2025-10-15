import numpy as np
from typing import List, Callable, Tuple, Dict
from functools import partial

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


def get_strain_stretch_edge2D3D(node0, node1, l_k):
    # Works for both 2D and 3D.
    # l_k (float): Reference (undeformed) length of the edge.
    edge = node1 - node0
    edgeLen = np.linalg.norm(edge)
    epsX = edgeLen / l_k - 1
    return epsX


def grad_and_hess_strain_stretch_edge3D_ZeroStrainStiff(node0, node1, l_k, tol=1e-10):
    '''
    Compute the gradient and Hessian of the axial stretch of a 3D edge with
    respect to the DOF vector (6 DOFs: x,y,z coords of the two nodes).

    Inputs:
      node0: length-3 array – position of the first node [x0,y0,z0]
      node1: length-3 array – position of the second node [x1,y1,z1]
      l_k:    float        – reference (undeformed) length of the edge

    Outputs:
      dF: length-6 array   – gradient of stretch w.r.t. [x0,y0,z0,x1,y1,z1]
      dJ: 6×6 array        – Hessian of stretch
    '''
    # edge vector and its length
    edge    = node1 - node0
    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen

    # axial stretch
    epsX = get_strain_stretch_edge2D3D(node0, node1, l_k)

    # gradient of stretch w.r.t. edge-vector
    dF_unit = tangent / l_k
    dF = np.zeros(6)
    dF[0:3] = -dF_unit
    dF[3:6] =  dF_unit

    # Hessian of squared-stretch w.r.t. edge-vector (3×3)
    I3 = np.eye(3)
    M  = 2.0 / l_k * (
           (1.0/l_k - 1.0/edgeLen) * I3
         + (1.0/edgeLen) * np.outer(edge, edge) / edgeLen**2
        )

    # convert to Hessian of stretch itself
    if abs(epsX) < tol:
        # small‐strain limit: (I - t t^T)/(L0 * L)
        M2 = (I3 - np.outer(tangent, tangent)) / (l_k * edgeLen)
    else:
        # full nonlinear Hessian of ε
        M2 = 1.0/(2.0*epsX) * (M - 2.0*np.outer(dF_unit, dF_unit))
        
    # assemble 6×6 Hessian
    dJ = np.zeros((6,6))
    dJ[ 0:3,  0:3] =  M2
    dJ[ 3:6,  3:6] =  M2
    dJ[ 0:3,  3:6] = -M2
    dJ[ 3:6,  0:3] = -M2

    return dF, dJ


def grad_and_hess_strain_stretch_edge3D(node0, node1, l_k):
    '''
    Compute the gradient and Hessian of the axial stretch of a 3D edge with
    respect to the DOF vector (6 DOFs: x,y,z coords of the two nodes).

    Inputs:
      node0: length-3 array – position of the first node [x0,y0,z0]
      node1: length-3 array – position of the second node [x1,y1,z1]
      l_k:    float        – reference (undeformed) length of the edge

    Outputs:
      dF: length-6 array   – gradient of stretch w.r.t. [x0,y0,z0,x1,y1,z1]
      dJ: 6×6 array        – Hessian of stretch
    '''
    # edge vector and its length
    edge    = node1 - node0
    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen

    # axial stretch
    epsX = get_strain_stretch_edge2D3D(node0, node1, l_k)

    # gradient of stretch w.r.t. edge-vector
    dF_unit = tangent / l_k
    dF = np.zeros(6)
    dF[0:3] = -dF_unit
    dF[3:6] =  dF_unit

    # Hessian of squared-stretch w.r.t. edge-vector (3×3)
    I3 = np.eye(3)
    M  = 2.0 / l_k * (
           (1.0/l_k - 1.0/edgeLen) * I3
         + (1.0/edgeLen) * np.outer(edge, edge) / edgeLen**2
        )

    # convert to Hessian of stretch itself
    if epsX == 0.0:
        M2 = np.zeros_like(M)
        
    else:
        M2 = 1.0/(2.0*epsX) * (M - 2.0*np.outer(dF_unit, dF_unit))

    # assemble 6×6 Hessian
    dJ = np.zeros((6,6))
    dJ[ 0:3,  0:3] =  M2
    dJ[ 3:6,  3:6] =  M2
    dJ[ 0:3,  3:6] = -M2
    dJ[ 3:6,  0:3] = -M2

    return dF, dJ


def fun_grad_hess_energy_stretch_linear_elastic_edge(node0, node1, l_0 = None, ks = None):
    # H has two terms, material elastic stiffness + geometric stiffness
    
    strain_stretch = get_strain_stretch_edge2D3D(node0, node1, l_0)
    G_strain, H_strain = grad_and_hess_strain_stretch_edge3D(node0, node1, l_0)

    gradE_strain = ks * strain_stretch * l_0
    hessE_strain = ks * l_0

    G = gradE_strain * G_strain
    H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)
    
    sub_block = H[0:3, 0:3]
    squared = sub_block**2
    sum_of_squares = np.sum(squared)
    # Verify stiffness with norm in small strain. With finite strain, geometric stiffness matters.
    print("Spring stiffness verify, node1=",node1)
    print("Sum of squares for 3DOFs from H:", sum_of_squares)
    print("squared ks/l0 :", (ks/l_0)**2)

    return G, H


def fun_grad_hess_energy_stretch_linear_elastic_edge_thermal(node0, node1, l_0 = None, ks = None, eps_th=0.0):
    # H has two terms, material elastic stiffness + geometric stiffness
    
    strain_stretch = get_strain_stretch_edge2D3D(node0, node1, l_0) - eps_th
    G_strain, H_strain = grad_and_hess_strain_stretch_edge3D(node0, node1, l_0)

    gradE_strain = ks * strain_stretch * l_0
    hessE_strain = ks * l_0

    G = gradE_strain * G_strain
    H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)
    
    sub_block = H[0:3, 0:3]
    squared = sub_block**2
    sum_of_squares = np.sum(squared)
    # # Verify stiffness with norm in small strain. With finite strain, geometric stiffness matters.
    # print("Spring stiffness verify, node1=",node1)
    # print("Sum of squares for 3DOFs from H:", sum_of_squares)
    # print("squared ks/l0 :", (ks/l_0)**2)

    return G, H


def assign_thermal_strains(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_thermal: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    inside: bool = True
) -> np.ndarray:
    """
    Assign thermal strain to each edge whose midpoint lies (inside/outside)
    the region defined by `region_fn`.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
        Coordinates of each node.
    connectivity : (Nedges,3) array
        Each row [eid, n0, n1].  We assume eid runs 0..Nedges‑1.
    eps_thermal : float
        Thermal strain to assign.
    region_fn : callable
        Given mids=(Nedges,3) array of midpoints, returns a boolean array
        of length Nedges: True where the edge should be “hot”.
    inside : bool
        If True, assign eps_thermal where region_fn is True;
        if False, assign where region_fn is False (i.e. outside).

    Returns
    -------
    epsilon_th : (Nedges,) array
        Thermal strain for each edge.
    """
    # 1) midpoints for every edge
    #    connectivity[:,1:3] are the two node‐indices per edge
    n0 = connectivity[:,1].astype(int)
    n1 = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # shape (Nedges,3)

    # 2) evaluate region test
    mask = region_fn(mids, node_xyz)  # boolean array length Nedges

    # 3) fill
    epsilon_th = np.zeros(connectivity.shape[0], dtype=float)
    if inside:
        epsilon_th[mask] = eps_thermal
    else:
        epsilon_th[~mask] = eps_thermal

    return epsilon_th


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
    (and vice‐versa if inside=False).

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
    do a linear ramp from eps_max at the mesh mid‐height line to eps_min at
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
      thermal strain on the mid‐height line (t=0)
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
      semi‐axes of the ellipse in x‐ and y‐directions
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


def rect_region(mids: np.ndarray, node_xyz) -> np.ndarray:
    x, y = mids[:,0], mids[:,1]
    return (x >= x_min) & (x <= x_max) \
        & (y >= y_min) & (y <= y_max)


def circle_region(mids: np.ndarray,
                  node_xyz) -> np.ndarray:
    """
    True for edge‑midpoints within the circle of radius^2=r2 about (cx,cy).
    """
    x_coords = node_xyz[:, 0]
    y_coords = node_xyz[:, 1]
    cx = 0.5 * (x_coords.min() + x_coords.max())
    cy = 0.5 * (y_coords.min() + y_coords.max())
    radius = 0.5
    r2 = radius**2 

    dx = mids[:, 0] - cx
    dy = mids[:, 1] - cy
    return (dx*dx + dy*dy) <= r2


def bowl_region(mids: np.ndarray, node_xyz) -> np.ndarray:
    """
    Return True for edge‐midpoints in a central disk OR in every other
    wedge of the surrounding ring, producing a starburst pattern.
    """
    # 1) compute mesh center once
    x_coords = node_xyz[:, 0]
    y_coords = node_xyz[:, 1]
    cx = 0.5 * (x_coords.min() + x_coords.max())
    cy = 0.5 * (y_coords.min() + y_coords.max())

    # 2) radial limits
    r_inner = 0.5   # radius of the solid central disk
    r_outer = r_inner * 5.0  # outer edge of starburst ring

    # 3) how many wedges?
    n_wedges    = 12
    wedge_angle = 2*np.pi / n_wedges

    # 4) compute polar coords of each midpoint
    dx = mids[:,0] - cx
    dy = mids[:,1] - cy
    r   = np.sqrt(dx*dx + dy*dy)
    θ   = np.mod(np.arctan2(dy, dx), 2*np.pi)

    # 5) mask for central disk
    in_center = (r <= r_inner)

    # 6) mask for ring
    in_ring = (r > r_inner) & (r <= r_outer)

    # 7) alternating wedges: even‐numbered sectors only
    sector_idx   = (θ // wedge_angle).astype(int)
    in_even_wedge = (sector_idx % 2) == 0

    # 8) combine: center OR alternating wedges in ring
    return in_center | (in_ring & in_even_wedge)


def circle_six_arms_region(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float],
    circle_radius: float,
    arm_half_width: float,
    arm_half_length: float
) -> np.ndarray:
    """
    Return True for edge‐midpoints lying inside:
      1) a small circle of radius `circle_radius` around `circle_center`, OR
      2) any of six rectangular "arms" of half-width `arm_half_width`
         and half-length `arm_half_length`, radiating at 0°, 60°, 120°, ….
    """
    x0, y0 = circle_center
    x, y   = mids[:,0] - x0, mids[:,1] - y0

    # 1) small circle mask
    in_circle = (x*x + y*y) <= circle_radius**2

    # 2) six arms at 60° increments
    arms = np.zeros_like(in_circle, dtype=bool)
    # angles for the arms
    phis = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for phi in phis:
        # unit vector along arm
        ux, uy = np.cos(phi), np.sin(phi)
        # projection onto the arm direction
        proj =  x*ux + y*uy
        # perpendicular distance from arm axis
        perp = np.abs(-x*uy + y*ux)
        # mask points within the rectangular arm
        arms |= (perp <= arm_half_width) & (proj >= -arm_half_length) & (proj <= arm_half_length)

    return in_circle | arms


def square_X_region(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float],
    circle_radius: float,
    arm_half_width: float,
    arm_half_length: float
) -> np.ndarray:
    """
    Return True for edge‐midpoints lying inside:
      1) a small circle of radius `circle_radius` around `circle_center`, OR
      2) two diagonal rectangular “arms” (an X shape) of half‐width
         `arm_half_width` and half‐length `arm_half_length` at ±45°.
    """
    # translate into local (X,Y) about the circle center
    cx, cy = circle_center
    X = mids[:,0] - cx
    Y = mids[:,1] - cy

    # 1) little central circle
    in_circle = (X*X + Y*Y) <= circle_radius**2

    # 2) two diagonal arms at φ=+45° and φ=−45°
    arms = np.zeros_like(in_circle, dtype=bool)
    for phi in (np.pi/4, -np.pi/4):
        ux, uy = np.cos(phi), np.sin(phi)
        # projection along the strip
        proj =  X*ux + Y*uy
        # signed distance perpendicular to the strip
        perp = np.abs(-X*uy + Y*ux)
        arms |= (
            (perp <= arm_half_width) &
            (proj >= -arm_half_length) &
            (proj <= +arm_half_length)
        )

    return in_circle | arms


def stripe_region(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float, float],
    circle_radius: float,
    cross_half_width: float,
    cross_half_length: float
) -> np.ndarray:
    
    x0, y0 = circle_center
    x, y   = mids[:,0], mids[:,1]

    # 1) small circle mask
    dx = x - x0
    dy = y - y0
    in_circle = (dx*dx + dy*dy) <= circle_radius**2

    # 2) horizontal bar mask: |y - y0| <= cross_half_width AND |x - x0| <= cross_half_length
    in_hbar = (np.abs(y - y0) <= cross_half_width) & (np.abs(x - x0) <= cross_half_length)

    # 3) vertical bar mask: |x - x0| <= cross_half_width AND |y - y0| <= cross_half_length
    in_vbar = (np.abs(x - x0) <= cross_half_width) & (np.abs(y - y0) <= cross_half_length)

    return  in_hbar


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


def two_half_rings_with_bar(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float] = (0.15, 0.08),
    separation: float = 0.01,
    inner_radius: float = 0.01,
    outer_radius: float = 0.05,
    bar_half_width: float = 0.002,
    wedge_angle: float = np.pi/6   # half‐angle of the missing wedge
) -> np.ndarray:
    """
    Return True for points inside two 'C'-shaped annuli (left+right)
    plus the rectangular bar joining their openings.

    The rings are full 360° annuli except for a wedge of angle 2*wedge_angle
    centered on the +x direction (for the left C) or -x direction (for the right C).
    """

    x0, y0 = circle_center
    # compute the two ring centers
    dx = separation/2
    xc_L, yc_L = x0 - dx, y0
    xc_R, yc_R = x0 + dx, y0

    x = mids[:,0]
    y = mids[:,1]

    # distances to centers
    dL = np.hypot(x - xc_L, y - yc_L)
    dR = np.hypot(x - xc_R, y - yc_R)

    # full annulus masks
    ann_L = (dL >= inner_radius) & (dL <= outer_radius)
    ann_R = (dR >= inner_radius) & (dR <= outer_radius)

    # compute angles
    thL = np.arctan2(y - yc_L, x - xc_L)   # in (-π, π]
    thR = np.arctan2(y - yc_R, x - xc_R)

    # for the left C, we remove the wedge around +x (th ≈ 0)
    keep_L = np.abs(thL) >= wedge_angle
    # but also only the “left” half‐plane x <= x0
    half_L = (x <= x0)

    # for the right C, remove the wedge around –x (th ≈ π or –π)
    # we shift angles to [–π, π] and remove |thR − π| < wedge_angle
    wrap_diff = np.mod(thR + np.pi, 2*np.pi) - np.pi  # shift into (–π, π]
    keep_R = np.abs(wrap_diff - np.pi) >= wedge_angle
    half_R = (x >= x0)

    mask_L = ann_L & keep_L & half_L
    mask_R = ann_R & keep_R & half_R

    # connecting bar between the two openings:
    #   y within +/- bar_half_width, x between the two ring centers
    bar_mask = (
        (y >= y0 - bar_half_width) &
        (y <= y0 + bar_half_width) &
        (x >= xc_L) &
        (x <= xc_R)
    )

    return mask_L | mask_R | bar_mask


def two_half_rings_with_spokes_and_bar(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float] = (0.15, 0.08),
    separation: float         = 0.10,
    inner_radius: float       = 0.04,
    outer_radius: float       = 0.08,
    arc_thickness: float      = 0.04,
    spoke_half_angle: float   = np.pi/180 * 3,  # 3° half-width
    bar_half_width: float     = 0.02
) -> np.ndarray:
    """
    Return True for points lying on the two 'C' shapes with:
      - inner & outer arcs
      - 6 radial spokes
      - connecting bar
    """
    x0, y0    = circle_center
    dx        = separation/2
    xc_L, yc_L = x0 - dx, y0
    xc_R, yc_R = x0 + dx, y0

    x = mids[:,0]
    y = mids[:,1]

    # 1) radial distances & basic annulus
    dL = np.hypot(x - xc_L, y - yc_L)
    dR = np.hypot(x - xc_R, y - yc_R)
    in_ann_L = (dL <= outer_radius)
    in_ann_R = (dR <= outer_radius)

    # 2) outer and inner arcs (thick), *with parentheses* around each comparison
    arc_L_outer = in_ann_L & (np.abs(dL - outer_radius) <= arc_thickness)
    arc_R_outer = in_ann_R & (np.abs(dR - outer_radius) <= arc_thickness)
    arc_L_inner = in_ann_L & (np.abs(dL - inner_radius) <= arc_thickness)
    arc_R_inner = in_ann_R & (np.abs(dR - inner_radius) <= arc_thickness)

    # restrict to half-annulus: left half for L, right half for R
    half_L = (x <= x0)
    half_R = (x >= x0)
    arc_L  = (arc_L_outer | arc_L_inner) & half_L
    arc_R  = (arc_R_outer | arc_R_inner) & half_R

    # 3) radial spokes (thin angular wedges), 6 spokes at 0°, 60°, …
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    thL    = np.arctan2(y - yc_L, x - xc_L)
    thR    = np.arctan2(y - yc_R, x - xc_R)

    def angular_diff(a, b):
        diff = (a - b + np.pi) % (2*np.pi) - np.pi
        return diff

    spoke_L = np.zeros_like(x, dtype=bool)
    spoke_R = np.zeros_like(x, dtype=bool)
    for phi in angles:
        spoke_L |= (
            in_ann_L & half_L &
            (np.abs(angular_diff(thL, phi)) <= spoke_half_angle)
        )
        spoke_R |= (
            in_ann_R & half_R &
            (np.abs(angular_diff(thR, phi)) <= spoke_half_angle)
        )

    # 4) connecting bar
    bar_mask = (
        (y >= y0 - bar_half_width) &
        (y <= y0 + bar_half_width) &
        (x >= xc_L) &
        (x <= xc_R)
    )

    return arc_L | arc_R | spoke_L | spoke_R | bar_mask


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


def white_C_pattern_mask(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float],
    separation: float,
    inner_radius: float,
    outer_radius: float,
    arc_thickness: float,
    n_spokes: int = 6,
    spoke_half_angle: float = np.deg2rad(3),
    bar_half_width: float
) -> np.ndarray:
    """
    Return a boolean mask of shape (Nedges,) that is True wherever
    an edge‐midpoint falls on the white 'C' pattern:
      - two half‐annuli (inner & outer arcs of thickness `arc_thickness`)
      - n_spokes radial spokes
      - a straight bar connecting the two openings.

    Parameters
    ----------
    mids : (Nedges,3) array of edge midpoints
    circle_center : (x0,y0) centerline of the two C’s
    separation    : distance between the two C‐centers along x
    inner_radius  : inner radius of each C
    outer_radius  : outer radius of each C
    arc_thickness : radial thickness of each arc (≈ line‐width/2)
    n_spokes      : how many spokes per C (default 6)
    spoke_half_angle : half‐angular width of each spoke (radians)
    bar_half_width   : half‐height of the joining bar
    """
    x0,y0 = circle_center
    dx    = separation/2
    # left‐ and right‐centers
    centers = [(x0-dx,y0), (x0+dx,y0)]
    x_mid,y_mid = x0,y0

    x = mids[:,0]
    y = mids[:,1]
    mask = np.zeros(len(mids), bool)

    # Build each C
    for (xc,yc) in centers:
        dx_ = x - xc
        dy_ = y - yc
        r  = np.hypot(dx_, dy_)

        # 1) thick arcs (inner & outer)
        arc_outer = (np.abs(r-outer_radius) <= arc_thickness)
        arc_inner = (np.abs(r-inner_radius) <= arc_thickness)
        # restrict to the correct half‐plane
        if xc < x_mid:
            halfplane = (x <= x_mid)
        else:
            halfplane = (x >= x_mid)
        mask |= (arc_outer | arc_inner) & halfplane

        # 2) radial spokes
        th = np.arctan2(dy_, dx_)   # in (−π,π]
        for k in range(n_spokes):
            phi = 2*np.pi * k / n_spokes
            diff = (th - phi + np.pi) % (2*np.pi) - np.pi
            spoke = (np.abs(diff) <= spoke_half_angle) \
                    & (r >= inner_radius) & (r <= outer_radius) \
                    & halfplane
            mask |= spoke

    # 3) straight bar between the two C‐openings
    xcL,ycL = centers[0]
    xcR,ycR = centers[1]
    bar = (
        (np.abs(y - y_mid) <= bar_half_width)
        & (x >= xcL) & (x <= xcR)
    )
    mask |= bar

    return mask


def white_star_pattern_mask(
    mids: np.ndarray,
    *,
    center: Tuple[float,float] = (0.2, 0.1),
    star_radius: float    = 0.05,
    star_thickness: float = 0.002,
    n_spokes: int         = 6,
    bar_length: float     = 0.15,
    bar_thickness: float  = 0.002
) -> np.ndarray:
    """
    Return a boolean mask for points lying on:
      - a 'star' of n_spokes equally spaced (60° apart if n_spokes=6),
        each spoke of length star_radius and thickness star_thickness,
      - plus a straight bar extending to the right of the center
        of length bar_length and thickeness bar_thickness.

    mids : (Nedges,3) array of edge‐midpoints
    center : (cx,cy) center of the star
    star_radius : how far each of the 6 spokes reaches
    star_thickness : half‐width of each spoke
    bar_length : length of the extra rightward arm
    bar_thickness : half‐width of that bar
    """
    cx, cy = center
    # translate to star‐centered coords
    X = mids[:,0] - cx
    Y = mids[:,1] - cy

    mask = np.zeros_like(X, dtype=bool)

    # 1) the six star‐spokes, every 360/n_spokes degrees
    phis = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)
    for phi in phis:
        ux, uy = np.cos(phi), np.sin(phi)
        # projection along the spoke:
        proj =  X*ux + Y*uy
        # perpendicular distance to the spoke‐axis:
        perp = np.abs(-X*uy + Y*ux)
        mask |= (proj >= 0) & (proj <= star_radius) & (perp <= star_thickness)

    # 2) the extra long rightward bar (phi=0)
    proj_bar = X           # since phi=0 ⇒ u=(1,0) ⇒ proj = X
    perp_bar = np.abs(Y)   # perpendicular distance is just |y|
    mask |= (proj_bar >= 0) & (proj_bar <= bar_length) & (perp_bar <= bar_thickness)

    return mask


def whole_peanut_pattern_mask(
    mids: np.ndarray,
    *,
    left_center:  Tuple[float,float],
    right_center: Tuple[float,float],
    star_radius:  float   = 0.05,
    star_thickness: float = 0.002,
    n_spokes:      int    = 6,
    beam_thickness: float = 0.002
) -> np.ndarray:
    """
    Return True for edge-midpoints lying on:
      - two 'star' shapes at left_center and right_center,
        each with n_spokes radial arms of length star_radius
        and half-width star_thickness;
      - plus a rectangular beam connecting the two centers,
        of half-width beam_thickness.

    mids            : (Nedges,3) array of midpoints
    left_center     : (xL,yL)
    right_center    : (xR,yR)
    star_radius     : radial length of each spoke
    star_thickness  : half-width of each spoke arm
    n_spokes        : number of arms per star
    beam_thickness  : half-width of the connecting beam
    """
    xL,yL = left_center
    xR,yR = right_center
    X = mids[:,0]
    Y = mids[:,1]
    mask = np.zeros_like(X, dtype=bool)

    # ---- helper to build one star at (cx,cy) ----
    def _star_mask(cx, cy):
        # translate to local coords
        dx = X - cx
        dy = Y - cy
        local = np.zeros_like(dx, dtype=bool)

        phis = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)
        for phi in phis:
            ux, uy = np.cos(phi), np.sin(phi)
            proj =  dx*ux + dy*uy
            perp = np.abs(-dx*uy + dy*ux)
            local |= (proj >= 0) & (proj <= star_radius) & (perp <= star_thickness)

        return local

    # draw left and right stars
    mask |= _star_mask(xL,yL)
    mask |= _star_mask(xR,yR)

    # ---- beam between the two star centers ----
    # restrict X between xL and xR (in either order)
    x_min, x_max = min(xL, xR), max(xL, xR)
    beam = (
        (X >= x_min) & (X <= x_max) &
        (np.abs(Y - ((yL+yR)/2)) <= beam_thickness)
    )
    mask |= beam

    return mask


def whole_peanut_region(
    mids: np.ndarray,
    *,
    center: Tuple[float, float],
    delta_shape: float      = 0.05,
    star_radius: float      = 0.05,
    star_thickness: float   = 0.002,
    n_spokes: int           = 6,
    beam_thickness: float   = 0.002
) -> np.ndarray:
    """
    Region‐mask for the “whole‑peanut” pattern:
      • two star‑shapes at (center_x ± delta_shape, center_y)
      • joined by a rectangular beam of half‑width beam_thickness.

    Parameters
    ----------
    mids : (Nedges,3) array  
        coordinates of edge‑midpoints
    center : (x_center, y_center)  
        midpoint between the two star centers
    delta_shape : float  
        half the distance between the star centers along x
    star_radius : float  
        length of each star’s radial arm
    star_thickness : float  
        half‑width of each arm
    n_spokes : int  
        number of arms per star
    beam_thickness : float  
        half‑width of the beam connecting the stars

    Returns
    -------
    mask : (Nedges,) boolean array  
        True where mids lie in the peanut region
    """
    x0, y0 = center
    left_center  = (x0 - delta_shape, y0)
    right_center = (x0 + delta_shape, y0)

    return whole_peanut_pattern_mask(
        mids,
        left_center   = left_center,
        right_center  = right_center,
        star_radius   = star_radius,
        star_thickness= star_thickness,
        n_spokes      = n_spokes,
        beam_thickness= beam_thickness
    )