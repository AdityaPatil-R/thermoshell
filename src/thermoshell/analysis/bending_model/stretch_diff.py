import numpy as np
from typing import List, Callable, Tuple, Dict
from elasticity import get_strain_stretch_edge2D3D, grad_and_hess_strain_stretch_edge3D

def fun_DEps_grad_hess(xloc: np.ndarray,
                       nodes: List[int],
                       edge_length_fn: Callable[[int,int], float]
                       ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    For one hinge‐quad with global nodes [n0,n1,oppA,oppB]:
      Δε = +ε(n0–oppA)
           −ε(n0–oppB)
           +ε(n1–oppA)
           −ε(n1–oppB)
    Returns (Deps, GDeps (12,), HDeps (12×12)) matching that exact sequence.
    """
    # unpack your four node‐IDs
    n0, n1, oppA, oppB = nodes

    # build pairs in the exact order you want:
    pairs = [
      (n0, oppA),
      (n0, oppB),
      (n1, oppA),
      (n1, oppB),
    ]
    signs = [+1,  -1,  +1,  -1]

    Deps  = 0.0
    GDeps = np.zeros(12)
    HDeps = np.zeros((12,12))

    for (a,b), s in zip(pairs, signs):
        # find their local slots in xloc
        ia = nodes.index(a)
        ib = nodes.index(b)
        loc0 = slice(3*ia,   3*ia+3)
        loc1 = slice(3*ib,   3*ib+3)

        x0 = xloc[loc0]
        x1 = xloc[loc1]
        L0 = edge_length_fn(a, b)

        # 1) scalar stretch and accumulate
        eps_ab = get_strain_stretch_edge2D3D(x0, x1, L0)
        Deps  += s * eps_ab

        # 2) its grad & hess
        dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)

        # 3) scatter into our 12-vector
        GDeps[loc0] +=  s * dG_e[0:3]
        GDeps[loc1] +=  s * dG_e[3:6]

        idx = list(range(3*ia,3*ia+3)) + list(range(3*ib,3*ib+3))
        HDeps[np.ix_(idx,idx)] +=  s * dH_e

    return Deps, GDeps, HDeps



def fun_DEps_grad_hess_thermal(xloc: np.ndarray,
                               nodes: List[int],
                               edge_length_fn: Callable[[int,int], float],
                               eps_th_vector: np.ndarray,
                               edge_dict: Dict[Tuple[int,int],int]
                               ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    For one hinge‐quad with global nodes [n0,n1,oppA,oppB]:
      Δε = +ε(n0–oppA)
           −ε(n0–oppB)
           +ε(n1–oppA)
           −ε(n1–oppB)
    Like fun_DEps_grad_hess, but for Δε_mech - Δε_th:
        Δε_total = Σ_signs [ ε_mech(a–b) - ε_th[eid(a,b)] ]
    Returns (Deps_total, GDeps, HDeps).
    """
    # unpack your four node‐IDs
    n0, n1, oppA, oppB = nodes

    # build pairs in the exact order you want:
    pairs = [
      (n0, oppA),
      (n0, oppB),
      (n1, oppA),
      (n1, oppB),
    ]
    signs = [+1,  -1,  +1,  -1]

    Deps  = 0.0
    GDeps = np.zeros(12)
    HDeps = np.zeros((12,12))

    for (a,b), s in zip(pairs, signs):
        # find their local slots in xloc
        ia = nodes.index(a)
        ib = nodes.index(b)
        loc0 = slice(3*ia,   3*ia+3)
        loc1 = slice(3*ib,   3*ib+3)

        x0 = xloc[loc0]
        x1 = xloc[loc1]
        L0 = edge_length_fn(a, b)

        # mechanical stretch
        eps_mech = get_strain_stretch_edge2D3D(x0, x1, L0)
        # look up the thermal strain for that same edge
        eid      = edge_dict[tuple(sorted((a,b)))]
        eps_th   = eps_th_vector[eid]

        # accumulate signed (mech - th)
        Deps += s * (eps_mech - eps_th)

        # 2) its grad & hess
        dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)

        # 3) scatter into our 12-vector
        GDeps[loc0] +=  s * dG_e[0:3]
        GDeps[loc1] +=  s * dG_e[3:6]

        idx = list(range(3*ia,3*ia+3)) + list(range(3*ib,3*ib+3))
        HDeps[np.ix_(idx,idx)] +=  s * dH_e

    return Deps, GDeps, HDeps