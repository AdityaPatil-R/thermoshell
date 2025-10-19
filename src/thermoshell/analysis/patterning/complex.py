import numpy as np
from typing import Tuple

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