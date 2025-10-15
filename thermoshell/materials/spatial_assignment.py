import numpy as np
from typing import Callable, Tuple, Dict
from dataclasses import replace
# For type checking only, modules will be imported by main.py
if typing.TYPE_CHECKING:
    from thermoshell.core.params import GeomParams, MaterialParams


# --- 1. Patterned Young's Modulus Assignment (Y_array) ---

def assign_spatial_properties(
    P_MAT: 'MaterialParams',
    P_GEOM: 'GeomParams',
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    region_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Builds the (Nedges,) array of Young's Moduli (Y_array) based on the geometric
    pattern defined by region_fn. Includes a radial ramp effect within the soft material.

    Parameters:
        P_MAT (MaterialParams): Material constants (Y_soft, Y_hard, Y_ratio, etc.).
        P_GEOM (GeomParams): Geometric constants (Pattern_center, OuterR, etc.).
        node_coords (np.ndarray): Nodal coordinates (Nnodes, 3).
        connectivity (np.ndarray): Edge connectivity [eid, n0, n1].
        region_fn (Callable): Function that returns True for midpoints inside the hard/active region.

    Returns:
        np.ndarray: Y_array (Nedges,) containing the effective Young's modulus for each edge.
    """
    Y_soft, Y_hard = P_MAT.youngs_soft, P_MAT.youngs_hard
    Y_ratio = P_MAT.youngs_ratio
    
    # 1. Compute midpoints
    n0 = connectivity[:, 1].astype(int)
    n1 = connectivity[:, 2].astype(int)
    mids = 0.5 * (node_coords[n0] + node_coords[n1])  # (Nedges, 3)

    # 2. Test region: mask is True where pattern is applied (hard material/active region)
    mask = region_fn(mids)

    # 3. Compute radial ramp distance and normalized radius r_norm in [0, 1]
    cx, cy = P_GEOM.pattern_center
    R = P_GEOM.pattern_radius

    dx = mids[:, 0] - cx
    dy = mids[:, 1] - cy
    r = np.sqrt(dx * dx + dy * dy)
    
    # Normalized radius [0..1], clipped at pattern_radius R
    r_norm = np.clip(r / R, 0.0, 1.0) 

    # 4. Define soft/hard moduli endpoints for the ramp
    # Modulus at center (r=0) is the base Y
    Y_base_r = np.where(mask, Y_hard, Y_soft)
    
    # Modulus at outer edge (r=R) is Y_ratio * Y
    Y_scaled_R = np.where(mask, Y_hard * Y_ratio, Y_soft * Y_ratio)

    # Calculate the ramp: Y(r) = Y_base_r + (Y_scaled_R - Y_base_r) * r_norm
    Y_array = Y_base_r + (Y_scaled_R - Y_base_r) * r_norm

    # The region is usually assigned Y_soft (low Y) and the outside is Y_hard (high Y).
    # This logic may need slight adjustment based on whether the region_fn marks 
    # the SOFT or HARD material depending on the specific pattern.
    
    # Default behavior: mask=HARD, ~mask=SOFT (assuming active regions are bilayer/hard)
    Y = np.empty_like(r)
    
    # Inside mask: Soft material (Y_soft) with ramp
    # Outside mask: Hard material (Y_hard) with ramp
    
    # Determine the ramp arrays based on where Y_soft or Y_hard should be applied
    soft_ramp = Y_soft + (Y_soft * Y_ratio - Y_soft) * r_norm
    hard_ramp = Y_hard + (Y_hard * Y_ratio - Y_hard) * r_norm
    
    # The simulation pattern has the active region (mask=True) as the soft/active material 
    # and the complement (~mask=False) as the inactive/hard material.
    
    # Let's assume the mask marks the hard, inactive material (bilayer) for the peanut example (iMesh=2)
    # The actual patterning logic will be handled by the specific region_fn. 
    # For now, we follow the original logic where the final Y is assigned based on the region_fn.
    
    # Reverting to the simpler version which assumes the Y_array already holds 
    # the correct base (Y_soft/Y_hard) and ramped values, provided the masks
    # are set up correctly in the thermal loading module.
    
    # The core logic here should use the correct soft/hard definitions:
    
    # Y_soft_r (base Y soft) + (Y_soft_R (scaled Y soft) - Y_soft_r) * r_norm
    # Y_hard_r (base Y hard) + (Y_hard_R (scaled Y hard) - Y_hard_r) * r_norm
    
    Y_soft_ramp = Y_soft + (Y_soft * Y_ratio - Y_soft) * r_norm
    Y_hard_ramp = Y_hard + (Y_hard * Y_ratio - Y_hard) * r_norm

    # If the mask is True, we assign Y_soft_ramp (active/bilayer is soft/active in original code)
    # If the mask is False, we assign Y_hard_ramp (passive/hard in original code)
    
    # NOTE: The original complex logic implies the *inactive* region is the hard part of the bilayer,
    # and the *active* region is the soft part. Based on the FE model using kb_bilayer for Y_hard regions.
    
    # If mask is True (active thermal strain region): This should be the soft material part
    Y[mask] = Y_soft_ramp[mask]
    
    # If mask is False (inactive/complement region): This should be the hard material part
    Y[~mask] = Y_hard_ramp[~mask]


    return Y


# --- 2. Discrete Stiffness Array Initialization ---

def initialize_stiffness_arrays(
    P_MAT: 'MaterialParams',
    P_GEOM: 'GeomParams',
    Y_array: np.ndarray,
    HingeQuads_order: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes the (Nedges,) ks_array and (Nhinges,) kb_array based on 
    the spatially assigned Y_array and scaling factors.

    Parameters:
        P_MAT (MaterialParams): Contains kb/ks_base constants and scaling factors.
        P_GEOM (GeomParams): Contains Y_soft, Y_hard for comparison.
        Y_array (np.ndarray): Spatially assigned Young's Modulus for each edge.
        HingeQuads_order (np.ndarray): Hinge definitions for indexing kb_array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (ks_array, kb_array)
    """
    Y_soft, Y_hard = P_MAT.youngs_soft, P_MAT.youngs_hard
    
    # --- Axial Stiffness (ks_array) ---
    
    # Map the spatially assigned Young's Modulus back to the discrete stiffness constants
    # The comparison relies on Y_array being near Y_soft or Y_hard (or their scaled versions)
    # We use the full hard/soft range for comparison, ignoring the ramp effect for mapping to ks_base:
    
    # 1. Identify "Hard" (bilayer) regions by modulus being closer to Y_hard
    is_hard = (Y_array >= (Y_soft + Y_hard) / 2.0)
    
    ks_array = np.where(
        is_hard,
        P_MAT.ks_bilayer_base, # Use bilayer ks for hard regions
        P_MAT.ks_soft_base     # Use soft ks for soft regions
    )
    
    # Apply global scaling factor
    ks_array = ks_array * P_MAT.factor_ks

    # --- Bending Stiffness (kb_array) ---
    
    hinge_eids = HingeQuads_order[:, 0]
    
    # Get the Young's Modulus only for the edges that are hinges
    Y_hinge = Y_array[hinge_eids]
    
    # Identify which hinges belong to a "Hard" region
    is_hard_hinge = (Y_hinge >= (Y_soft + Y_hard) / 2.0)
    
    # Map to the corresponding base bending stiffness
    kb_array = np.where(
        is_hard_hinge,
        P_MAT.kb_bilayer_base, # Use bilayer kb for hard/bilayer regions
        P_MAT.kb_soft_base     # Use soft kb for soft/single-layer regions
    )
    
    # Apply global scaling factor
    kb_array = kb_array * P_MAT.factor_kb
    
    return ks_array, kb_array
