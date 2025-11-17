import numpy as np
# Import the required local functions
from src.ThermalDES.bending_model.geometry import getTheta, calculate_stretch_difference_grad_hess, calculate_stretch_difference_grad_hess_thermal
from src.ThermalDES.assembly.assemblers import ElasticGHEdgesCoupled, ElasticGHEdgesCoupledThermal, ElasticGHEdges

from src.ThermalDES.material.unit_laws import (fun_grad_hess_energy_stretch_linear_elastic_edge, 
                                         fun_grad_hess_energy_stretch_linear_elastic_edge_thermal,
                                         get_strain_stretch_edge2D3D, 
                                         grad_and_hess_strain_stretch_edge3D)

from src.ThermalDES.bending_model.geometry import (getTheta, 
                                             gradTheta, 
                                             hessTheta)

from src.ThermalDES.bending_model.energy import (calculate_pure_bending_grad_hess, 
                                           calculate_coupled_bending_grad_hess, 
                                           calculate_coupled_bending_grad_hess_thermal)

from src.ThermalDES.assembly.energy_check import fun_total_system_energy_coupled_thermal


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

def test_ElasticGHEdges(energy_choice,theta_bar,Nedges,NP_total,Ndofs,ConnectivityMatrix_line,L0, 
                        X0, iPrint, HingeQuads_order ):
    
    q = X0.copy()
    Y  = 1.0
    h  = 0.1
    EA = 1.0; #Y * np.pi * h**2
    EI = 1.0; #Y * np.pi / 4 * h**4   # unused in this test
    model_choice = 1            # linear‐elastic stretch only
    
    struct1 = ElasticGHEdges(
        energy_choice = energy_choice,
        Nedges        = Nedges,
        NP_total      = NP_total,
        Ndofs         = Ndofs,
        connectivity  = ConnectivityMatrix_line,
        l0_ref        = L0,
        ks            = EA,
        hinge_quads   = HingeQuads_order,
        theta_bar     = theta_bar,
        kb            = 1.0,
        h             = h,
        model_choice  = 1
    )
    
    G, H = struct1.computeGradientHessian(q)

    # (6) Print results
    if iPrint:
        np.set_printoptions(precision=4, suppress=True)
        print("Assembled force vector G:")
        print(G)
        print("\nAssembled stiffness matrix H:")
        print(H)