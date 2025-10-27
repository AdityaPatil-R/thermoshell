import numpy as np
from typing import List, Callable, Tuple, Dict
from src.thermoshell.material.unit_laws import get_strain_stretch_edge2D3D, grad_and_hess_strain_stretch_edge3D

def signedAngle(u=None,
                v=None,
                n=None):
    # This function calculates the signed angle between two vectors, "u" and "v"
    # using an optional axis vector "n" to determine the direction of the angle.
    #
    # Parameters:
    #   u: numpy array-like, shape (3,), the first vector.
    #   v: numpy array-like, shape (3,), the second vector.
    #   n: numpy array-like, shape (3,), the axis vector that defines the plane
    #      in which the angle is measured. It determines the sign of the angle.
    #
    # Returns:
    #   angle: float, the signed angle (in radians) from vector "u" to vector "v".
    #          The angle is positive if the rotation from "u" to "v" follows
    #          the right-hand rule with respect to the axis "n", and negative otherwise.
    #
    # The function works by:
    # 1. Computing the cross product "w" of "u" and "v" to find the vector orthogonal
    #    to both "u" and "v".
    # 2. Calculating the angle between "u" and "v" using the arctan2 function, which
    #    returns the angle based on the norm of "w" (magnitude of the cross product)
    #    and the dot product of "u" and "v".
    # 3. Using the dot product of "n" and "w" to determine the sign of the angle.
    #    If this dot product is negative, the angle is adjusted to be negative.
    #
    # Example:
    #   signedAngle(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    #   This would return a positive angle (π/2 radians), as the rotation
    #   from the x-axis to the y-axis is counterclockwise when viewed along the z-axis.

    w = np.cross(u,v)
    dot_product = np.dot(u, v)
    angle = np.arctan2(np.linalg.norm(w), dot_product)

    if (np.dot(n,w) < 0):
        return -angle
    else:
        return angle


def mmt(matrix):
    return matrix + matrix.T


def getTheta(x0, 
             x1=None, 
             x2=None, 
             x3=None):
    if np.size(x0) == 12:  # Allow input where x0 contains all the info
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[0:3]

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0

    n0 = np.cross(m_e0, m_e1)
    n1 = np.cross(m_e2, m_e0)

    # Calculate the signed angle
    theta = signedAngle(n0, n1, m_e0)

    return theta


def gradTheta(x0, 
              x1 = None, 
              x2 = None, 
              x3 = None):
    if np.size(x0) == 12:  # Allow another type of input where x0 contains all the info
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[0:3]

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0
    m_e3 = x2 - x1
    m_e4 = x3 - x1

    m_cosA1 =  np.dot(m_e0, m_e1) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_cosA2 =  np.dot(m_e0, m_e2) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_cosA3 = -np.dot(m_e0, m_e3) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_cosA4 = -np.dot(m_e0, m_e4) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_sinA1 =  np.linalg.norm(np.cross(m_e0, m_e1)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_sinA2 =  np.linalg.norm(np.cross(m_e0, m_e2)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_sinA3 = -np.linalg.norm(np.cross(m_e0, m_e3)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_sinA4 = -np.linalg.norm(np.cross(m_e0, m_e4)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_nn1  =  np.cross(m_e0, m_e3)
    m_nn1 /=  np.linalg.norm(m_nn1)
    m_nn2  = -np.cross(m_e0, m_e4)
    m_nn2 /=  np.linalg.norm(m_nn2)

    m_h1  =  np.linalg.norm(m_e0) * m_sinA1
    m_h2  =  np.linalg.norm(m_e0) * m_sinA2
    m_h3  = -np.linalg.norm(m_e0) * m_sinA3 
    m_h4  = -np.linalg.norm(m_e0) * m_sinA4 
    m_h01 =  np.linalg.norm(m_e1) * m_sinA1
    m_h02 =  np.linalg.norm(m_e2) * m_sinA2

    # Initialize the gradient
    gradTheta = np.zeros(12)

    gradTheta[0:3]  = m_cosA3 * m_nn1 / m_h3 + m_cosA4 * m_nn2 / m_h4
    gradTheta[3:6]  = m_cosA1 * m_nn1 / m_h1 + m_cosA2 * m_nn2 / m_h2
    gradTheta[6:9]  = -m_nn1 / m_h01
    gradTheta[9:12] = -m_nn2 / m_h02

    return gradTheta

def hessTheta(x0, 
              x1=None, 
              x2=None, 
              x3=None):
    if np.size(x0) == 12: 
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[0:3]

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0
    m_e3 = x2 - x1
    m_e4 = x3 - x1

    m_cosA1 =  np.dot(m_e0, m_e1) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_cosA2 =  np.dot(m_e0, m_e2) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_cosA3 = -np.dot(m_e0, m_e3) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_cosA4 = -np.dot(m_e0, m_e4) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_sinA1 =  np.linalg.norm(np.cross(m_e0, m_e1)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_sinA2 =  np.linalg.norm(np.cross(m_e0, m_e2)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_sinA3 = -np.linalg.norm(np.cross(m_e0, m_e3)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_sinA4 = -np.linalg.norm(np.cross(m_e0, m_e4)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_nn1  =  np.cross(m_e0, m_e3)
    m_nn1 /=  np.linalg.norm(m_nn1)
    m_nn2  = -np.cross(m_e0, m_e4)
    m_nn2 /=  np.linalg.norm(m_nn2)

    m_h1  =  np.linalg.norm(m_e0) * m_sinA1
    m_h2  =  np.linalg.norm(m_e0) * m_sinA2
    m_h3  = -np.linalg.norm(m_e0) * m_sinA3
    m_h4  = -np.linalg.norm(m_e0) * m_sinA4
    m_h01 =  np.linalg.norm(m_e1) * m_sinA1
    m_h02 =  np.linalg.norm(m_e2) * m_sinA2

    # Gradient of Theta
    grad_theta = np.zeros((12, 1))
    grad_theta[0:3]  = (m_cosA3 * m_nn1 / m_h3 + m_cosA4 * m_nn2 / m_h4).reshape(-1, 1)
    grad_theta[3:6]  = (m_cosA1 * m_nn1 / m_h1 + m_cosA2 * m_nn2 / m_h2).reshape(-1, 1)
    grad_theta[6:9]  = (-m_nn1 / m_h01).reshape(-1, 1)
    grad_theta[9:12] = (-m_nn2 / m_h02).reshape(-1, 1)

    # Intermediate matrices for Hessian
    m_m1  =  np.cross(m_nn1, m_e1) / np.linalg.norm(m_e1)
    m_m2  = -np.cross(m_nn2, m_e2) / np.linalg.norm(m_e2)
    m_m3  = -np.cross(m_nn1, m_e3) / np.linalg.norm(m_e3)
    m_m4  =  np.cross(m_nn2, m_e4) / np.linalg.norm(m_e4)
    m_m01 = -np.cross(m_nn1, m_e0) / np.linalg.norm(m_e0)
    m_m02 =  np.cross(m_nn2, m_e0) / np.linalg.norm(m_e0)

    # Hessian matrix components
    M331  = m_cosA3 / (m_h3 ** 2)    * np.outer(m_m3, m_nn1)
    M311  = m_cosA3 / (m_h3 * m_h1)  * np.outer(m_m1, m_nn1)
    M131  = m_cosA1 / (m_h1 * m_h3)  * np.outer(m_m3, m_nn1)
    M3011 = m_cosA3 / (m_h3 * m_h01) * np.outer(m_m01, m_nn1)
    M111  = m_cosA1 / (m_h1 ** 2)    * np.outer(m_m1, m_nn1)
    M1011 = m_cosA1 / (m_h1 * m_h01) * np.outer(m_m01, m_nn1)

    M442  = m_cosA4 / (m_h4 ** 2)    * np.outer(m_m4, m_nn2)
    M422  = m_cosA4 / (m_h4 * m_h2)  * np.outer(m_m2, m_nn2)
    M242  = m_cosA2 / (m_h2 * m_h4)  * np.outer(m_m4, m_nn2)
    M4022 = m_cosA4 / (m_h4 * m_h02) * np.outer(m_m02, m_nn2)
    M222  = m_cosA2 / (m_h2 ** 2)    * np.outer(m_m2, m_nn2)
    M2022 = m_cosA2 / (m_h2 * m_h02) * np.outer(m_m02, m_nn2)

    B1 = 1 / np.linalg.norm(m_e0) ** 2 * np.outer(m_nn1, m_m01)
    B2 = 1 / np.linalg.norm(m_e0) ** 2 * np.outer(m_nn2, m_m02)

    N13  = 1 / (m_h01 * m_h3) * np.outer(m_nn1, m_m3)
    N24  = 1 / (m_h02 * m_h4) * np.outer(m_nn2, m_m4)
    N11  = 1 / (m_h01 * m_h1) * np.outer(m_nn1, m_m1)
    N22  = 1 / (m_h02 * m_h2) * np.outer(m_nn2, m_m2)
    N101 = 1 / (m_h01 ** 2)   * np.outer(m_nn1, m_m01)
    N202 = 1 / (m_h02 ** 2)   * np.outer(m_nn2, m_m02)

    # Initialize Hessian of Theta
    hess_theta = np.zeros((12, 12))

    hess_theta[0:3, 0:3]   = mmt(M331) - B1 + mmt(M442) - B2
    hess_theta[0:3, 3:6]   = M311 + M131.T + B1 + M422 + M242.T + B2
    hess_theta[0:3, 6:9]   = M3011 - N13
    hess_theta[0:3, 9:12]  = M4022 - N24
    hess_theta[3:6, 3:6]   = mmt(M111) - B1 + mmt(M222) - B2
    hess_theta[3:6, 6:9]   = M1011 - N11
    hess_theta[3:6, 9:12]  = M2022 - N22
    hess_theta[6:9, 6:9]   = -mmt(N101)
    hess_theta[9:12, 9:12] = -mmt(N202)

    # Make Hessian symmetric
    hess_theta[3:6, 0:3]  = hess_theta[0:3, 3:6].T
    hess_theta[6:9, 0:3]  = hess_theta[0:3, 6:9].T
    hess_theta[9:12, 0:3] = hess_theta[0:3, 9:12].T
    hess_theta[6:9, 3:6]  = hess_theta[3:6, 6:9].T
    hess_theta[9:12, 3:6] = hess_theta[3:6, 9:12].T

    return hess_theta

def calculate_stretch_difference_grad_hess(xloc: np.ndarray,
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

def calculate_stretch_difference_grad_hess_thermal(xloc: np.ndarray,
                               nodes: List[int],
                               edge_length_fn: Callable[[int, int], float],
                               eps_th_vector: np.ndarray,
                               edge_dict: Dict[Tuple[int, int], int]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    For one hinge-quad with global nodes [n0,n1,oppA,oppB]:
      Δε = +ε(n0-oppA)−ε(n0-oppB)+ε(n1-oppA)−ε(n1-oppB)
    Like calculate_stretch_difference_grad_hess, but for Δε_mech - Δε_th:
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