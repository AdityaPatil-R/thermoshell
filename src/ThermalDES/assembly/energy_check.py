import numpy as np
# Import the required local functions
from src.ThermalDES.material.unit_laws import get_strain_stretch_edge2D3D
from src.ThermalDES.bending_model.geometry import getTheta, calculate_stretch_difference_grad_hess, calculate_stretch_difference_grad_hess_thermal
from src.ThermalDES.assembly.assemblers import ElasticGHEdgesCoupled, ElasticGHEdgesCoupledThermal

def fun_total_system_energy_coupled(
        q: np.ndarray,
        model: ElasticGHEdgesCoupled
        ) -> float:
    """
    Compute the total energy E(q) = Σ_edges ½ k_s ε^2 ℓ
                       + Σ_hinges ½ k_b [θ - β Δε]^2
    using exactly the same loops as your assembler.

    Parameters
    ----------
    q : (ndof,) array
      The current flattened nodal coordinate vector.
    model : ElasticGHEdgesCoupled
      Your energy object, with .connectivity, .hinge_quads, .ks, .kb, .beta, .l_ref, etc.

    Returns
    -------
    E : float
      The scalar total energy.
    """
    E = 0.0
    # 1) stretch
    for eid, n0, n1 in model.connectivity:
        idx0 = slice(3*n0,   3*n0+3)
        idx1 = slice(3*n1,   3*n1+3)
        x0, x1 = q[idx0], q[idx1]
        L0     = model.l_ref[eid]
        eps    = get_strain_stretch_edge2D3D(x0, x1, L0)
        ke     = model.ks_array[eid]    # pick this edge’s axial stiffness
        E     += 0.5 * ke * eps*eps * L0

    # 2) coupled bending
    for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(model.hinge_quads):
        nodes = [n0, n1, oppA, oppB]
        inds  = sum([[3*n+i for i in range(3)] for n in nodes], [])
        xloc  = q[inds]

        θ, gθ, Hθ = None, None, None
        # just need θ and Δε here
        θ   = getTheta(xloc)
        # compute Δε exactly as in _assemble_bending_coupled
        Deps, _, _ = calculate_stretch_difference_grad_hess(
            xloc, nodes, model._edge_length
        )

        kb_i = model.kb_array[h_idx]
        diff = θ - model.beta * Deps
        E   += 0.5 * kb_i * diff*diff

    return E


def fun_total_system_energy_coupled_thermal(
        q: np.ndarray,
        model: ElasticGHEdgesCoupledThermal
        ) -> float:
    """
    Compute the total energy E(q) = Σ_edges ½ k_s ε^2 ℓ
                       + Σ_hinges ½ k_b [θ - β Δε]^2
    using exactly the same loops as your assembler.

    Parameters
    ----------
    q : (ndof,) array
      The current flattened nodal coordinate vector.
    model : ElasticGHEdgesCoupled
      Your energy object, with .connectivity, .hinge_quads, .ks, .kb, .beta, .l_ref, etc.

    Returns
    -------
    E : float
      The scalar total energy.
    """
    E = 0.0
    # 1) stretch
    for eid, n0, n1 in model.connectivity:
        idx0 = slice(3*n0,   3*n0+3)
        idx1 = slice(3*n1,   3*n1+3)
        x0, x1 = q[idx0], q[idx1]
        L0     = model.l_ref[eid]
        ke     = model.ks_array[eid]
        eps    = get_strain_stretch_edge2D3D(x0, x1, L0) - model.eps_th[eid]
        # eps    = get_strain_stretch_edge2D3D(x0, x1, L0)
        E     += 0.5 * ke * eps*eps * L0

    # 2) coupled bending
    for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(model.hinge_quads):
        nodes = [n0, n1, oppA, oppB]
        inds  = sum([[3*n+i for i in range(3)] for n in nodes], [])
        xloc  = q[inds]

        θ, gθ, Hθ = None, None, None
        # just need θ and Δε here
        θ   = getTheta(xloc)
        # Deps, _, _ = calculate_stretch_difference_grad_hess(xloc, nodes, model._edge_length)
        Deps, _, _ = calculate_stretch_difference_grad_hess_thermal(xloc, nodes, model._edge_length, model.eps_th, model.edge_dict)

        kb_i = model.kb_array[h_idx]
        diff = θ - model.beta * Deps
        E   += 0.5 * kb_i * diff*diff

    return E