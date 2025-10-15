



def FDM_ElasticGHEdges(energy_choice,Nedges,NP_total,Ndofs,connectivity,l0_ref, 
                       X0, iPrint, HingeQuads_order, theta_bar):
    
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
        connectivity  = connectivity,
        l0_ref        = l0_ref,
        ks            = EA,
        hinge_quads   = HingeQuads_order,
        theta_bar     = theta_bar,
        kb            = 1.0,
        h             = EI,
        model_choice  = 1
    )
        
    
    F_elastic, J_elastic = struct1.computeGradientHessian(q)
    change = 1.0e-5
    J_fdm = np.zeros((Ndofs,Ndofs))
    for c in range(Ndofs):
        q_plus = q.copy()
        q_plus[c] += change
        F_change, _ = struct1.computeGradientHessian(q_plus)
        J_fdm[:,c] = (F_change - F_elastic) / change
    
    # Plot: Hessians
    if iPrint:
        plt.figure(10)
        plt.plot(J_elastic.flatten(), 'ro', label='Analytical')  # Flatten and plot analytical Hessian
        plt.plot(J_fdm.flatten(), 'b^', label='Finite Difference')
        plt.legend(loc='best')
        plt.title('Verify hess of Ebend wrt x')
        plt.xlabel('Index')
        plt.ylabel('Hessian')
        plt.show()
        error = np.linalg.norm(J_elastic - J_fdm) / np.linalg.norm(J_elastic)
        print("FD Hessian check for ElasticGHEdges")
        print("Hessian error:")
        print("Relative Frobenius‐norm error:", error)
        max_err = np.max(np.abs(J_elastic - J_fdm))
        rms_err = np.linalg.norm(J_elastic - J_fdm) / np.sqrt(J_elastic.size)
        print(f"L_inf max abs error: {max_err:.4e},\nRoot-mean-square (RMS) error: {rms_err:.4e}")
    return


def FDM_ElasticGHEdgesCoupled(
    energy_choice:int,
    Nedges:int,
    NP_total:int,
    Ndofs:int,
    connectivity:np.ndarray,
    l0_ref:np.ndarray,
    hinge_quads:np.ndarray,
    theta_bar:float,
    ks:float,
    kb:float,
    beta:float,
    h:float,
    model_choice:int,
    X0:np.ndarray,
    iPrint:bool=True,
    eps:float=1e-6
):
    """
    Finite‐difference check of ∇²E for ElasticGHEdgesCoupled.
    """
    # initial q
    q = X0.copy()

    # build the coupled model
    struct = ElasticGHEdgesCoupled(
        energy_choice=energy_choice,
        Nedges=Nedges,
        NP_total=NP_total,
        Ndofs=Ndofs,
        connectivity=connectivity,
        l0_ref=l0_ref,
        ks=ks,
        hinge_quads=hinge_quads,
        theta_bar=theta_bar,
        kb=kb,
        h=h,
        beta=beta,
        model_choice=model_choice
    )

    # analytic force & stiffness
    G0, H0 = struct.computeGradientHessian(q)

    # finite‐difference Hessian
    H_fd = np.zeros((Ndofs,Ndofs))
    for c in range(Ndofs):
        q_plus = q.copy()
        q_plus[c] += eps
        Gp, _ = struct.computeGradientHessian(q_plus)
        H_fd[:,c] = (Gp - G0)/eps

    if iPrint:
        # plot comparison
        plt.figure(figsize=(8,4))
        plt.plot(H0.flatten(), 'ro', label='analytic')
        plt.plot(H_fd.flatten(), 'b^', label='FD')
        plt.legend(loc='best')
        plt.xlabel('Hessian entry index')
        plt.ylabel('∂²E/∂xᵢ∂xⱼ')
        plt.title('ElasticGHEdgesCoupled: analytic vs FD Hessian')
        plt.tight_layout()
        plt.show()

        # norms
        rel = np.linalg.norm(H0 - H_fd)/np.linalg.norm(H0)
        linf = np.max(np.abs(H0 - H_fd))
        rms  = np.linalg.norm(H0 - H_fd)/np.sqrt(H0.size)
        print("FD Hessian check for ElasticGHEdgesCoupled")
        print(f"  relative Frobenius error: {rel:.3e}")
        print(f"  L_inf max abs error:      {linf:.3e}")
        print(f"  RMS error:                {rms:.3e}")

    return


def FDM_ElasticGHEdgesCoupledThermal(
    energy_choice:int,
    Nedges:int,
    NP_total:int,
    Ndofs:int,
    connectivity:np.ndarray,
    l0_ref:np.ndarray,
    hinge_quads:np.ndarray,
    theta_bar:float,
    ks:float,
    kb:float,
    beta:float,
    h:float,
    eps_th_vector:np.ndarray,
    model_choice:int,
    X0:np.ndarray,
    iPrint:bool=True,
    eps:float=1e-6
):
    """
    Finite‐difference check of ∇²E for ElasticGHEdgesCoupled.
    """
    # initial q
    q = X0.copy()

    # build the coupled model
    struct = ElasticGHEdgesCoupledThermal(
        energy_choice=energy_choice,
        Nedges=Nedges,
        NP_total=NP_total,
        Ndofs=Ndofs,
        connectivity=connectivity,
        l0_ref=l0_ref,
        ks=ks,
        hinge_quads=hinge_quads,
        theta_bar=theta_bar,
        kb=kb,
        beta=beta,
        epsilon_th=eps_th_vector,
        model_choice=model_choice
    )

    # analytic force & stiffness
    G0, H0 = struct.computeGradientHessian(q)

    # finite‐difference Hessian
    H_fd = np.zeros((Ndofs,Ndofs))
    for c in range(Ndofs):
        q_plus = q.copy()
        q_plus[c] += eps
        Gp, _ = struct.computeGradientHessian(q_plus)
        H_fd[:,c] = (Gp - G0)/eps

    if iPrint:
        # plot comparison
        plt.figure(figsize=(8,4))
        plt.plot(H0.flatten(), 'ro', label='analytic')
        plt.plot(H_fd.flatten(), 'b^', label='FD')
        plt.legend(loc='best')
        plt.xlabel('Hessian entry index')
        plt.ylabel('∂²E/∂xᵢ∂xⱼ')
        plt.title('ElasticGHEdgesCoupledThermal: analytic vs FD Hessian')
        plt.tight_layout()
        plt.show()

        # norms
        rel = np.linalg.norm(H0 - H_fd)/np.linalg.norm(H0)
        linf = np.max(np.abs(H0 - H_fd))
        rms  = np.linalg.norm(H0 - H_fd)/np.sqrt(H0.size)
        print("FD Hessian check for ElasticGHEdgesCoupledThermal")
        print(f"  relative Frobenius error: {rel:.3e}")
        print(f"  L_inf max abs error:      {linf:.3e}")
        print(f"  RMS error:                {rms:.3e}")

    return