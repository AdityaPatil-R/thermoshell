import numpy as np
from typing import Tuple

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

