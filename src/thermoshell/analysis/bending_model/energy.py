import numpy as np
from typing import List, Callable, Tuple, Dict
from .geometry import getTheta, gradTheta, hessTheta 
from .stretch_diff import fun_DEps_grad_hess, fun_DEps_grad_hess_thermal

def getEb_Shell(x0, x1=None, x2=None, x3=None, theta_bar=0, kb=1):
    """
    Compute the bending energy for a shell.

    Returns:
    E (scalar): Bending energy.
    """
    # Allow another type of input where x0 contains all the information
    if np.size(x0) == 12:
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[:3]

    # Compute theta, gradient, and Hessian
    theta = getTheta(x0, x1, x2, x3)  # Replace with your getTheta function in Python
    grad = gradTheta(x0, x1, x2, x3)  # Replace with your gradTheta function in Python

    # E = 0.5 * kb * (theta-thetaBar)^2
    E = 0.5 * kb * (theta - theta_bar) ** 2

    return E



def gradEb_hessEb_Shell(x0, x1=None, x2=None, x3=None, theta_bar=0, kb=1):
    """
    Compute the gradient and Hessian of the bending energy for a shell.

    Parameters:
    x0 (array): Can either be a 3-element array (single point) or a 12-element array.
    x1, x2, x3 (arrays): Optional, 3-element arrays specifying points.
    theta_bar (float): Reference angle.
    kb (float): Bending stiffness.

    Returns:
    dF (array): Gradient of the bending energy.
    dJ (array): Hessian of the bending energy.
    """
    # Allow another type of input where x0 contains all the information
    if np.size(x0) == 12:
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[:3]

    # Compute theta, gradient, and Hessian
    theta = getTheta(x0, x1, x2, x3)  # Replace with your getTheta function in Python
    grad = gradTheta(x0, x1, x2, x3)  # Replace with your gradTheta function in Python

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = dE/dx = 2 * (theta-thetaBar) * gradTheta
    dF = 0.5 * kb * (2 * (theta - theta_bar) * grad)

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = 0.5 * kb * (2 (theta-thetaBar) d theta/dx)
    # J = dF/dx = 0.5 * kb * [ 2 (d theta / dx) transpose(d theta/dx) +
    #       2 (theta-thetaBar) (d^2 theta/ dx^2 ) ]
    hess = hessTheta(x0, x1, x2, x3)  # Replace with your hessTheta function in Python
    dJ = 0.5 * kb * (2 * np.outer(grad, grad) + 2 * (theta - theta_bar) * hess)

    return dF, dJ


def fun_coupled_Ebend_grad_hess(
        xloc: np.ndarray,
        nodes: List[int],
        edge_length_fn: Callable[[int,int], float],
        beta: float,
        kb: float
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    For one hinge‑quad with 4 local nodes (in the order [n0,n1,oppA,oppB]) and 
    xloc.shape==(12,), compute the gradient & Hessian of
      ½·kb·[θ − β·Δε]²
    where Δε = +ε(n0–oppA)
              −ε(n0–oppB)
              +ε(n1–oppA)
              −ε(n1–oppB).
    Returns:
      dG (12,) : ∂E/∂xloc
      dH (12,12) : ∂²E/∂xloc²
    """
    # 1) pure‐bending pieces
    θ   = getTheta(xloc)
    gθ  = gradTheta(xloc)
    Hθ  = hessTheta(xloc)

    # 2) the Δε, its grad & hess in the exact ordering you want
    Deps, GDeps, HDeps = fun_DEps_grad_hess(xloc, nodes, edge_length_fn)

    # 3) form force‐like and stiffness‐like pieces
    f_h = θ - beta * Deps           # scalar
    C_h = gθ  - beta * GDeps        # shape (12,)

    dG = kb * f_h * C_h
    dH = kb * ( np.outer(C_h, C_h)
              + f_h*(Hθ - beta*HDeps) )

    return dG, dH


def fun_coupled_Ebend_grad_hess_thermal(
        xloc: np.ndarray,
        nodes: List[int],
        edge_length_fn: Callable[[int,int], float],
        beta: float,
        kb: float,
        eps_th: np.ndarray,
        edge_dict: Dict[Tuple[int,int],int]
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    For one hinge‑quad with 4 local nodes (in the order [n0,n1,oppA,oppB]) and 
    xloc.shape==(12,), compute the gradient & Hessian of
      ½·kb·[θ − β·Δε]²
    where Δε = +ε(n0–oppA)
              −ε(n0–oppB)
              +ε(n1–oppA)
              −ε(n1–oppB).
    Change kb according to dihedral angle sign.
    Returns:
      dG (12,) : ∂E/∂xloc
      dH (12,12) : ∂²E/∂xloc²
    """
    # 1) pure‐bending pieces
    θ   = getTheta(xloc)
    gθ  = gradTheta(xloc)
    Hθ  = hessTheta(xloc)

    #  # Not converge for 1.1 when ks_hard=100*ks_soft
    #  # converge for 1.05
    Factor_kb = 1.0
    kb_angle  = Factor_kb*kb if θ > 0 else kb
    
    # 2) the Δε, its grad & hess in the exact ordering you want
    Deps, GDeps, HDeps = fun_DEps_grad_hess_thermal(xloc, nodes, edge_length_fn, eps_th, edge_dict)

    # 3) form force‐like and stiffness‐like pieces
    f_h = θ - beta * Deps           # scalar
    C_h = gθ  - beta * GDeps        # shape (12,)

    dG = kb_angle * f_h * C_h
    dH = kb_angle * ( np.outer(C_h, C_h)
                    + f_h*(Hθ - beta*HDeps) )

    return dG, dH