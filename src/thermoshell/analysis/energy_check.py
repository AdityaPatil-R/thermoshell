import numpy as np
# Import the required local functions
from analysis.material.unit_laws import get_strain_stretch_edge2D3D
from analysis.bending_model.geometry import getTheta
from analysis.bending_model.stretch_diff import fun_DEps_grad_hess, fun_DEps_grad_hess_thermal
from assembly.assemblers import ElasticGHEdgesCoupled, ElasticGHEdgesCoupledThermal

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
        Deps, _, _ = fun_DEps_grad_hess(
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
        # Deps, _, _ = fun_DEps_grad_hess(xloc, nodes, model._edge_length)
        Deps, _, _ = fun_DEps_grad_hess_thermal(xloc, nodes, model._edge_length, model.eps_th, model.edge_dict)

        kb_i = model.kb_array[h_idx]
        diff = θ - model.beta * Deps
        E   += 0.5 * kb_i * diff*diff

    return E

def test_fun_total_energy_grad_hess_thermal(
        plt,
        model: ElasticGHEdgesCoupledThermal,
        q0: np.ndarray,
        iplot: int = 1,
        eps: float = 1e-6
    ) -> None:
    """
    Finite‐difference check of ∇E and ∇²E for the full coupled energy.

    Prints out relative Frobenius‐norm, L_inf, RMS errors, and—if plot=True—draws
    comparisons of analytic vs FD for both gradient and Hessian entries.
    """
    nd = model.ndof

    # --- analytic ---
    G0, H0 = model.computeGradientHessian(q0)
    
    # --- FD gradient via central differences ---
    G_fd = np.zeros(nd)
    for i in range(nd):
        dq = np.zeros_like(q0); dq[i] = eps
        Ep = fun_total_system_energy_coupled_thermal(q0 + dq, model)
        Em = fun_total_system_energy_coupled_thermal(q0 - dq, model)
        G_fd[i] = (Ep - Em) / (2*eps)

    # --- FD Hessian via forward diff of gradient ---
    H_fd = np.zeros((nd, nd))
    for j in range(nd):
        dq = np.zeros_like(q0); dq[j] = eps
        Gp, _ = model.computeGradientHessian(q0 + dq)
        H_fd[:, j] = (Gp - G0) / eps

    # --- reporting ---
    def report(A, B, name):
        rel = np.linalg.norm(A - B) / np.linalg.norm(A)
        inf = np.max(np.abs(A - B))
        rms = np.linalg.norm(A - B) / np.sqrt(A.size)
        print(f"{name} error:")
        print(f"  relative Frobenius = {rel:.3e}")
        print(f"  L_inf max abs     = {inf:.3e}")
        print(f"  RMS               = {rms:.3e}\n")

    print("test_total_energy_grad_hess")
    print("=== total‐energy gradient check ===")
    report(G0, G_fd, "∇E")

    print("=== total‐energy Hessian check ===")
    report(H0, H_fd, "∇²E")

    # --- plotting ---
    iplot=1
    if iplot==1:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(G0,     'ro', label='analytic')
        plt.plot(G_fd,   'b.', label='FD')
        plt.title('Gradient ∇E of whole system w thermal strains')
        plt.xlabel('DOF index')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(H0.flatten(), 'ro', label='analytic')
        plt.plot(H_fd.flatten(), 'b.', label='FD')
        plt.title('Hessian ∇²E of whole system w thermal strains')
        plt.xlabel('entry index')
        plt.legend()

        plt.tight_layout()
        plt.show()