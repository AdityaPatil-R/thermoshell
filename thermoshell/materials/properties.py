import numpy as np
import typing
from dataclasses import dataclass, replace

# Assuming GeomParams and MaterialParams are available for type hinting, 
# although they'd typically be imported from core.params.
if typing.TYPE_CHECKING:
    from thermoshell.core.params import GeomParams, MaterialParams


# --- 1. Effective Flexural Rigidity for Bilayer Beam ---

def bilayer_flexural_rigidity(h_soft: float, h_hard: float, Y_soft: float, Y_hard: float, b1: float = 1.0, b2: float = 1.0) -> float:
    """
    Computes the effective flexural rigidity D_eff (per unit width) 
    for a symmetric bilayer composite beam using parallel axis theorem.
    
    This is used to derive the bending stiffness kb.

    Parameters:
        h_soft (float): Thickness of the soft material layer.
        h_hard (float): Thickness of the hard material layer.
        Y_soft (float): Young's modulus of the soft layer.
        Y_hard (float): Young's modulus of the hard layer.
        b1 (float): Width of soft layer (usually 1.0 for unit width).
        b2 (float): Width of hard layer (usually 1.0 for unit width).

    Returns:
        float: The total effective flexural rigidity D_eff (N*m).
    """
    # 1. Individual layer centroids (measured from the bottom of layer 1/soft layer)
    y1 = h_soft / 2.0
    y2 = h_soft + h_hard / 2.0

    # 2. Neutral axis location (y_bar)
    num = Y_soft * b1 * h_soft * y1 + Y_hard * b2 * h_hard * y2
    den = Y_soft * b1 * h_soft      + Y_hard * b2 * h_hard
    ybar = num / den

    # 3. Individual layer contributions to D (using parallel axis theorem)
    # D = I_self + A * d^2
    # Area moments of inertia (I_self = b*h^3 / 12)
    I1 = b1 * (h_soft**3) / 12.0
    I2 = b2 * (h_hard**3) / 12.0
    
    # Distance from neutral axis squared (d^2)
    d1_sq = (y1 - ybar)**2
    d2_sq = (y2 - ybar)**2
    
    D1 = Y_soft * (I1 + b1 * h_soft * d1_sq)
    D2 = Y_hard * (I2 + b2 * h_hard * d2_sq)

    # 4. Total effective rigidity
    D_eff = D1 + D2

    return D_eff


# --- 2. Geometric Parameters for Meshes ---

# Note: In a production environment, this data would ideally be stored in a 
# separate, non-Python configuration file (e.g., YAML, JSON) and loaded by core.params.
# For now, it is defined here as it is tightly coupled with the mesh ID (iMesh).
PARAMS_GEOM = {
    # (mean edge length lk, soft thickness h_soft, hard thickness h_hard)
    # These values were derived from the original geometry and calibration.
    1: (0.0032, 0.3e-3, 0.7e-3),  # iMesh 1: Circle
    2: (0.0040, 0.3e-3, 0.6e-3),  # iMesh 2: Rectangle/Peanut
    3: (0.0058, 0.3e-3, 1.0e-3),  # iMesh 3: Square
}


def derive_bilayer_constants(P_GEOM: 'GeomParams', P_MAT: 'MaterialParams') -> 'MaterialParams':
    """
    Calculates the discrete element stiffnesses (ks, kb) based on the 
    chosen mesh geometry (P_GEOM.mesh_id) and material properties (P_MAT).
    
    These constants are stored back in MaterialParams.

    Parameters:
        P_GEOM (GeomParams): Geometric configuration.
        P_MAT (MaterialParams): Base material moduli (Y_soft, Y_hard, etc.).

    Returns:
        MaterialParams: Updated MaterialParams dataclass with derived constants.
    """
    try:
        # Load geometric constants specific to the mesh ID
        lk, h_soft, h_hard = PARAMS_GEOM[P_GEOM.mesh_id]
    except KeyError:
        raise ValueError(f"Unsupported mesh ID: {P_GEOM.mesh_id!r}") from None

    Y_soft = P_MAT.youngs_soft
    Y_hard = P_MAT.youngs_hard
    
    # --- 1. Effective Bending Stiffness (kb) ---
    
    # Flexural rigidity D for the soft-only layer (used for pure soft regions)
    D_soft = Y_soft * (1.0) * (h_soft**3) / 12.0
    # Flexural rigidity D for the bilayer composite (used for hard/bilayer regions)
    D_bilayer = bilayer_flexural_rigidity(h_soft, h_hard, Y_soft, Y_hard)

    # Discrete Bending Stiffness kb = (2 / sqrt(3)) * D * width_factor
    # Assuming width factor is 1.0 for these discrete elements.
    kb_soft = (2.0 / np.sqrt(3.0)) * D_soft
    kb_bilayer = (2.0 / np.sqrt(3.0)) * D_bilayer

    # --- 2. Effective Axial Stiffness (ks) ---
    
    # Axial stiffness ks = Y * Area_eff. Effective Area = h * lk^2 * sqrt(3) / 2
    ks_soft = Y_soft * h_soft * (lk**2) * np.sqrt(3) / 2.0
    ks_hard = Y_hard * h_hard * (lk**2) * np.sqrt(3) / 2.0
    
    # Axial stiffness for the bilayer composite is the sum of the two layers
    ks_bilayer = ks_soft + ks_hard

    # --- 3. Update GeomParams (optional but useful) and MaterialParams ---
    
    # Update GeomParams fields that were looked up from the dict
    P_GEOM_updated = replace(P_GEOM, 
        thickness_soft=h_soft, 
        thickness_hard=h_hard,
        ref_length_lk=lk
    )
    
    # Update MaterialParams with the calculated discrete constants
    P_MAT_updated = replace(P_MAT,
        kb_soft_base=kb_soft,
        kb_bilayer_base=kb_bilayer,
        ks_soft_base=ks_soft,
        ks_bilayer_base=ks_bilayer
    )
    
    # Return the updated dataclasses
    return P_GEOM_updated, P_MAT_updated
