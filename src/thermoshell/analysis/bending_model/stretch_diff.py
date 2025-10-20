import numpy as np
from typing import List, Callable, Tuple, Dict
from analysis.material.unit_laws import get_strain_stretch_edge2D3D, grad_and_hess_strain_stretch_edge3D

def fun_DEps_grad_hess(xloc: np.ndarray,
                       nodes: List[int],
                       edge_length_fn: Callable[[int, int], float]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    For one hinge-quad with global nodes [n0,n1,oppA,oppB]:
      Δε = +ε(n0-oppA)-ε(n0-oppB)+ε(n1-oppA)-ε(n1-oppB)
    Returns (Deps, GDeps (12,), HDeps (12x12)) matching that exact sequence.
    """

    # Unpack the four node‐IDs
    n0, n1, oppA, oppB = nodes

    # Build pairs in the desired order
    pairs = [(n0, oppA), (n0, oppB), (n1, oppA), (n1, oppB)]
    signs = [        +1,         -1,         +1,         -1]

    Deps  = 0.0
    GDeps = np.zeros(12)
    HDeps = np.zeros((12, 12))

    for (a, b), s in zip(pairs, signs):
        # Find their local slots in xloc
        ia = nodes.index(a)
        ib = nodes.index(b)
        loc0 = slice(3*ia, 3*ia+3)
        loc1 = slice(3*ib, 3*ib+3)

        x0 = xloc[loc0]
        x1 = xloc[loc1]
        L0 = edge_length_fn(a, b)

        # Scalar stretch and accumulate
        eps_ab = get_strain_stretch_edge2D3D(x0, x1, L0)
        Deps  += s * eps_ab

        # Gradient & Hessian
        dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)

        # Scatter into the 12-vector
        GDeps[loc0] += s * dG_e[0:3]
        GDeps[loc1] += s * dG_e[3:6]

        idx = list(range(3*ia, 3*ia+3)) + list(range(3*ib, 3*ib+3))
        HDeps[np.ix_(idx, idx)] +=  s * dH_e

    return Deps, GDeps, HDeps

def fun_DEps_grad_hess_thermal(xloc: np.ndarray,
                               nodes: List[int],
                               edge_length_fn: Callable[[int, int], float],
                               eps_th_vector: np.ndarray,
                               edge_dict: Dict[Tuple[int, int], int]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    For one hinge-quad with global nodes [n0,n1,oppA,oppB]:
      Δε = +ε(n0-oppA)−ε(n0-oppB)+ε(n1-oppA)−ε(n1-oppB)
    Like fun_DEps_grad_hess, but for Δε_mech - Δε_th:
        Δε_total = Σ_signs [ε_mech(a-b) - ε_th[eid(a, b)]]
    Returns (Deps_total, GDeps, HDeps).
    """

    # Unpack the four node‐IDs
    n0, n1, oppA, oppB = nodes

    # Build pairs in the desired order
    pairs = [(n0, oppA), (n0, oppB), (n1, oppA), (n1, oppB)]
    signs = [        +1,         -1,         +1,         -1]

    Deps  = 0.0
    GDeps = np.zeros(12)
    HDeps = np.zeros((12, 12))

    for (a, b), s in zip(pairs, signs):
        # Find their local slots in xloc
        ia = nodes.index(a)
        ib = nodes.index(b)
        loc0 = slice(3*ia, 3*ia+3)
        loc1 = slice(3*ib, 3*ib+3)

        x0 = xloc[loc0]
        x1 = xloc[loc1]
        L0 = edge_length_fn(a, b)

        # Mechanical stretch
        eps_mech = get_strain_stretch_edge2D3D(x0, x1, L0)

        # Look up the thermal strain for that same edge
        eid    = edge_dict[tuple(sorted((a, b)))]
        eps_th = eps_th_vector[eid]

        # Accumulate signed (mech - th)
        Deps += s * (eps_mech - eps_th)

        # Gradient & Hessian
        dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)

        # Scatter into the 12-vector
        GDeps[loc0] += s * dG_e[0:3]
        GDeps[loc1] += s * dG_e[3:6]

        idx = list(range(3*ia, 3*ia+3)) + list(range(3*ib, 3*ib+3))
        HDeps[np.ix_(idx,idx)] +=  s * dH_e

    return Deps, GDeps, HDeps