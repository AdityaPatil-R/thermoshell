
# %% Def. classes: Construct equ of motion and solve with Newton iteration.

# understand the class timeStepper. Done.
# Figure out the problem of the solution. Fix z since z stiffness =0.


class timeStepper3D:
    def __init__(self, massVector, dt, qtol, maxIter, g, boundaryCondition, elasticModel, X0):
        """
        Implicit Newton time‐stepper for 3D DOFs.
        massVector        : 1D array of length ndof = 3*Nnodes
        dt                : time step size
        qtol              : Newton convergence tolerance on Δq
        maxIter           : max Newton iterations
        g                 : 3‐element gravity vector [gx,gy,gz]
        boundaryCondition : BoundaryConditions3D instance
        elasticModel      : object with computeGradientHessian(q)->(G,H)
        """
        self.massVector   = np.asarray(massVector, float)
        self.ndof         = len(self.massVector)
        if self.ndof % 3 != 0:
            raise ValueError("ndof must be a multiple of 3 for 3 DOFs/node")
        self.N            = self.ndof // 3

        self.dt           = float(dt)
        self.qtol         = float(qtol)
        self.maxIter      = int(maxIter)
        self.g            = np.asarray(g, float)
        if self.g.shape != (3,):
            raise ValueError("Gravity vector must have length 3")

        # Precompute weight Fg = M * g per node
        self.Fg           = np.zeros(self.ndof)
        self.makeWeight()

        # Diagonal mass matrix
        self.massMatrix   = np.zeros((self.ndof, self.ndof))
        self.makeMassMatrix()

        # Attach BC manager and elastic-energy model
        self.bc           = boundaryCondition
        self.elasticModel = elasticModel
        self.X0 = np.asarray(X0, float).copy()  # length = ndof
        
    def makeMassMatrix(self):
        """Build diagonal mass matrix M."""
        np.fill_diagonal(self.massMatrix, self.massVector)
        
        # for i in range(self.ndof):
        #     self.massMatrix[i,i] = self.massVector[i]

    def makeWeight(self):
        """Compute gravity‐induced forces per DOF."""
        for i in range(self.N):
            sl = slice(3*i, 3*i+3)
            self.Fg[sl] = self.massVector[sl] * self.g

    def beforeTimeStep(self) -> None:
        # Take care of any business that should be done BEFORE a time step.
        pass

    def afterTimeStep(self) -> None:
        # Take care of any business that should be done AFTER a time step.
        return

    def simulate(self, q_guess, q_old, u_old, a_old):
        """
        One implicit Newton step.
        Inputs:
          q_guess : initial guess for q_new (ndof,)
          q_old   : previous step displacements (ndof,)
          u_old   : previous step velocities (ndof,)
          a_old   : previous step accelerations (ndof,)
        Returns:
          q_new, u_new, a_new, flag (bool success)
        """
        dt = self.dt
        M  = self.massVector
        Fg = self.Fg
        X0 = self.X0

        # 1) Initialize and impose Dirichlet BCs
        q_new = q_guess.copy()
        for dof, u_val in zip(self.bc.fixedIndices, self.bc.fixedDOFs):
            q_new[dof] = X0[dof] + u_val
            
        # for k in range(len(bc.fixedIndices)):
        #     q_new[bc.fixedIndices[k]] = bc.fixedDOFs[k]

        # 2) Newton‐Raphson loop
        error = np.inf
        for iteration in range(1, self.maxIter+1):
            # internal: gradient & Hessian
            gradE, hessE = self.elasticModel.computeGradientHessian(q_new)
            # inertia: M*((q_new - q_old)/dt - u_old)/dt
            inertiaF = (M/dt) * (((q_new - q_old)/dt) - u_old)
            # residual & Jacobian
            R = inertiaF + gradE - Fg
            J = (self.massMatrix/dt**2) + hessE
            
            # restrict to freeIndex DOFs
            freeIndex = self.bc.freeIndices
            Rf   = R[freeIndex]
            Jf   = J[np.ix_(freeIndex, freeIndex)]
            dqf  = np.linalg.solve(Jf, Rf)
            
            # update free DOFs
            q_new[freeIndex] -= dqf
            
            error = np.linalg.norm(dqf)
            # print(f"  Newton iter={iteration}: error={error:.4e}")
            if error < self.qtol:
                break
            
        # new velocities and accelerations
        u_new = (q_new - q_old)/dt
        a_new = (inertiaF + gradE - Fg)/M
        return q_new, u_new, a_new, (error < self.qtol)
        


class timeStepper3D_static:
    def __init__(self, massVector, dt, qtol, maxIter, g, boundaryCondition, elasticModel, X0):
        """
        Implicit Newton time‐stepper for 3D DOFs.
        massVector        : 1D array of length ndof = 3*Nnodes
        dt                : time step size
        qtol              : Newton convergence tolerance on Δq
        maxIter           : max Newton iterations
        g                 : 3‐element gravity vector [gx,gy,gz]
        boundaryCondition : BoundaryConditions3D instance
        elasticModel      : object with computeGradientHessian(q)->(G,H)
        """
        self.massVector   = np.asarray(massVector, float)
        self.ndof         = len(self.massVector)
        if self.ndof % 3 != 0:
            raise ValueError("ndof must be a multiple of 3 for 3 DOFs/node")
        self.N            = self.ndof // 3

        self.dt           = float(dt)
        self.qtol         = float(qtol)
        self.maxIter      = int(maxIter)
        self.g            = np.asarray(g, float)
        if self.g.shape != (3,):
            raise ValueError("Gravity vector must have length 3")

        # Precompute weight Fg = M * g per node
        self.Fg           = np.zeros(self.ndof)
        self.makeWeight()

        # Diagonal mass matrix
        self.massMatrix   = np.zeros((self.ndof, self.ndof))
        self.makeMassMatrix()

        # Attach BC manager and elastic-energy model
        self.bc           = boundaryCondition
        self.elasticModel = elasticModel
        self.X0 = np.asarray(X0, float).copy()  # length = ndof
        self.last_num_iters = 0
        
    def makeMassMatrix(self):
        """Build diagonal mass matrix M."""
        np.fill_diagonal(self.massMatrix, self.massVector)
        
        # for i in range(self.ndof):
        #     self.massMatrix[i,i] = self.massVector[i]

    def makeWeight(self):
        """Compute gravity‐induced forces per DOF."""
        for i in range(self.N):
            sl = slice(3*i, 3*i+3)
            self.Fg[sl] = self.massVector[sl] * self.g

    def beforeTimeStep(self) -> None:
        # Take care of any business that should be done BEFORE a time step.
        pass

    def afterTimeStep(self) -> None:
        # Take care of any business that should be done AFTER a time step.
        return

    def simulate(self, q_guess, q_old, u_old, a_old):
        """
        One implicit Newton step.
        Inputs:
          q_guess : initial guess for q_new (ndof,)
          q_old   : previous step displacements (ndof,)
          u_old   : previous step velocities (ndof,)
          a_old   : previous step accelerations (ndof,)
        Returns:
          q_new, u_new, a_new, flag (bool success)
        """
        dt = self.dt
        M  = self.massVector
        Fg = self.Fg
        X0 = self.X0

        # 1) Initialize and impose Dirichlet BCs
        q_new = q_guess.copy()
        for dof, u_val in zip(self.bc.fixedIndices, self.bc.fixedDOFs):
            q_new[dof] = X0[dof] + u_val
            
        # for k in range(len(bc.fixedIndices)):
        #     q_new[bc.fixedIndices[k]] = bc.fixedDOFs[k]

        # 2) Newton‐Raphson loop
        rel_error = np.inf
        for iteration in range(1, self.maxIter+1):
            # internal: gradient & Hessian
            gradE, hessE = self.elasticModel.computeGradientHessian(q_new)
            # inertia: M*((q_new - q_old)/dt - u_old)/dt
            # inertiaF = (M/dt) * (((q_new - q_old)/dt) - u_old)
            # residual & Jacobian
            R = gradE - Fg
            J = hessE
            # R = inertiaF + gradE - Fg
            # J = (self.massMatrix/dt**2) + hessE
            
            # restrict to freeIndex DOFs
            freeIndex = self.bc.freeIndices
            Rf   = R[freeIndex]
            Jf   = J[np.ix_(freeIndex, freeIndex)]
            dqf  = np.linalg.solve(Jf, Rf)
            
            
            qfree = q_new[freeIndex]
            rel_error = np.linalg.norm(dqf) / max(0.1, np.linalg.norm(qfree))
            
            
            # update free DOFs
            q_new[freeIndex] -= dqf
            
            if rel_error < self.qtol:
                self.last_num_iters = iteration
                break
            
        else:
            self.last_num_iters = self.maxIter
            
        # new velocities and accelerations
        # u_new = (q_new - q_old)/dt
        # a_new = (inertiaF + gradE - Fg)/M
        # return q_new, u_new, a_new, (error < self.qtol)
        return q_new, (rel_error < self.qtol)
    


class timeStepper3D_static_gravity:
    def __init__(self, massVector, dt, qtol, maxIter, g, boundaryCondition, elasticModel, X0):
        """
        Implicit Newton time‐stepper for 3D DOFs.
        massVector        : 1D array of length ndof = 3*Nnodes
        dt                : time step size
        qtol              : Newton convergence tolerance on Δq
        maxIter           : max Newton iterations
        g                 : 3‐element gravity vector [gx,gy,gz]
        boundaryCondition : BoundaryConditions3D instance
        elasticModel      : object with computeGradientHessian(q)->(G,H)
        """
        self.massVector   = np.asarray(massVector, float)
        self.ndof         = len(self.massVector)
        if self.ndof % 3 != 0:
            raise ValueError("ndof must be a multiple of 3 for 3 DOFs/node")
        self.N            = self.ndof // 3

        self.dt           = float(dt)
        self.qtol         = float(qtol)
        self.maxIter      = int(maxIter)
        self.g            = np.asarray(g, float)
        if self.g.shape != (3,):
            raise ValueError("Gravity vector must have length 3")

        # Precompute weight Fg = M * g per node
        self.Fg           = np.zeros(self.ndof)
        self.makeWeight()

        # Diagonal mass matrix
        self.massMatrix   = np.zeros((self.ndof, self.ndof))
        self.makeMassMatrix()

        # Attach BC manager and elastic-energy model
        self.bc           = boundaryCondition
        self.elasticModel = elasticModel
        self.X0 = np.asarray(X0, float).copy()  # length = ndof
        
    def makeMassMatrix(self):
        """Build diagonal mass matrix M."""
        np.fill_diagonal(self.massMatrix, self.massVector)
        
        # for i in range(self.ndof):
        #     self.massMatrix[i,i] = self.massVector[i]

    def makeWeight(self):
        """Compute gravity‐induced forces per DOF."""
        for i in range(self.N):
            sl = slice(3*i, 3*i+3)
            self.Fg[sl] = self.massVector[sl] * self.g

    def beforeTimeStep(self) -> None:
        # Take care of any business that should be done BEFORE a time step.
        pass

    def afterTimeStep(self) -> None:
        # Take care of any business that should be done AFTER a time step.
        return

    def simulate(self, q_guess, q_old, u_old, a_old):
        """
        One implicit Newton step.
        Inputs:
          q_guess : initial guess for q_new (ndof,)
          q_old   : previous step displacements (ndof,)
          u_old   : previous step velocities (ndof,)
          a_old   : previous step accelerations (ndof,)
        Returns:
          q_new, u_new, a_new, flag (bool success)
        """
        dt = self.dt
        M  = self.massVector
        Fg = self.Fg
        X0 = self.X0

        # 1) Initialize and impose Dirichlet BCs
        q_new = q_guess.copy()
        for dof, u_val in zip(self.bc.fixedIndices, self.bc.fixedDOFs):
            q_new[dof] = X0[dof] + u_val
            
        # for k in range(len(bc.fixedIndices)):
        #     q_new[bc.fixedIndices[k]] = bc.fixedDOFs[k]

        # 2) Newton‐Raphson loop
        error = np.inf
        for iteration in range(1, self.maxIter+1):
            # internal: gradient & Hessian
            gradE, hessE = self.elasticModel.computeGradientHessian(q_new)
            # inertia: M*((q_new - q_old)/dt - u_old)/dt
            inertiaF = (M/dt) * (((q_new - q_old)/dt) - u_old)
            # residual & Jacobian
            R = gradE - Fg
            J = hessE
            # R = inertiaF + gradE - Fg
            # J = (self.massMatrix/dt**2) + hessE
            
            # restrict to freeIndex DOFs
            freeIndex = self.bc.freeIndices
            Rf   = R[freeIndex]
            Jf   = J[np.ix_(freeIndex, freeIndex)]
            dqf  = np.linalg.solve(Jf, Rf)
            
            # update free DOFs
            q_new[freeIndex] -= dqf
            
            error = np.linalg.norm(dqf)
            # print(f"  Newton iter={iteration}: error={error:.4e}")
            if error < self.qtol:
                break
            
        # new velocities and accelerations
        # u_new = (q_new - q_old)/dt
        # a_new = (inertiaF + gradE - Fg)/M
        # return q_new, u_new, a_new, (error < self.qtol)
        return q_new, (error < self.qtol)
    


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


def test_fun_total_energy_grad_hess(
        model: ElasticGHEdgesCoupled,
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
        Ep = fun_total_system_energy_coupled(q0 + dq, model)
        Em = fun_total_system_energy_coupled(q0 - dq, model)
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
        plt.title('Gradient ∇E of whole system')
        plt.xlabel('DOF index')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(H0.flatten(), 'ro', label='analytic')
        plt.plot(H_fd.flatten(), 'b.', label='FD')
        plt.title('Hessian ∇²E of whole system')
        plt.xlabel('entry index')
        plt.legend()

        plt.tight_layout()
        plt.show()


def test_fun_total_energy_grad_hess_thermal(
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


# %% Def. functions: Record variable values with loading history

def record_step(
    step: int,
    q_new: np.ndarray,
    elastic_model,
    connectivity: np.ndarray,
    L0: np.ndarray,
    ks_array: np.ndarray,
    hinge_quads: np.ndarray,
    Q_history: np.ndarray,
    R_history: np.ndarray,
    length_history: np.ndarray,
    strain_history: np.ndarray,
    stress_history: np.ndarray,
    theta_history: np.ndarray
):
    """
    Record displacements, reactions, strains, stresses, and dihedral angles
    at time‐step `step`.
    ----------
    step : int
      time‐step index (0…n_steps)
    q_new : (ndof,) array
      converged DOF vector at this step
    elastic_model : object
      must implement computeGradientHessian(q)->(gradE, hessE)
    connectivity : (Nedges,3) int‐array
      each row [eid, n0, n1]
    L0 : (Nedges,) array
      reference (undeformed) lengths per edge
    ks : float
      axial stiffness
    hinge_quads : (Nhinges,5) int‐array
      each row [eid, n0, n1, oppA, oppB]
    Q_history : (n_steps+1,ndof) array
      displacement history
    R_history : (n_steps+1,ndof) array
      reaction (gradient) history
    length_history : (n_steps+1,Nedges) array
      current edge lengths history
    strain_history : (n_steps+1,Nedges) array
      axial strain history
    stress_history : (n_steps+1,Nedges) array
      axial stress history
    theta_history : (n_steps+1,Nhinges) array
      dihedral angle history
    """
    # 1) record nodal displacements
    Q_history[step] = q_new

    # 2) record reaction = gradient of total energy
    gradE, _       = elastic_model.computeGradientHessian(q_new)
    R_history[step] = gradE

    # 3) record edge strains & stresses
    for i, edge in enumerate(connectivity):
        _, n0, n1 = edge
        p0 = q_new[3*n0 : 3*n0+3]
        p1 = q_new[3*n1 : 3*n1+3]
        L_current = np.linalg.norm(p1 - p0)
        length_history[step, i] = L_current
        eps = get_strain_stretch_edge2D3D(p0, p1, L0[i])
        strain_history[step, i] = eps
        ke = ks_array[i]
        stress_history[step, i] = ke * eps

    # 4) record dihedral angle at each hinge
    for j, (_, n0, n1, oppA, oppB) in enumerate(hinge_quads):
        # gather the four node coordinates
        x0 = q_new[3*n0  : 3*n0+3]
        x1 = q_new[3*n1  : 3*n1+3]
        x2 = q_new[3*oppA: 3*oppA+3]
        x3 = q_new[3*oppB: 3*oppB+3]
        # compute signed dihedral
        theta = getTheta(x0, x1, x2, x3)
        theta_history[step, j] = theta


def record_step_old(
    step: int,
    q_new: np.ndarray,
    elastic_model,
    connectivity: np.ndarray,
    L0: np.ndarray,
    EA: float,
    Q_history: np.ndarray,
    R_history: np.ndarray,
    strain_history: np.ndarray,
    stress_history: np.ndarray
):
    """
    Record displacements, reactions, strains and stresses at time‐step `step`.

    Parameters
    ----------
    step : int
      time‐step index (0…n_steps)
    q_new : (ndof,) array
      converged DOF vector at this step
    elastic_model : object
      must implement computeGradientHessian(q)->(gradE, hessE)
    connectivity : (Nedges,3) int‐array
      each row [eid, n0, n1]
    L0 : (Nedges,) array
      reference lengths per edge
    EA : float
      axial stiffness = E·A
    Q_history, R_history : (n_steps+1,ndof) arrays
      pre‐allocated
    strain_history, stress_history : (n_steps+1,Nedges) arrays
      pre‐allocated
    """
    # 1) displacement
    Q_history[step] = q_new

    # 2) reaction = gradient of elastic energy
    gradE, _       = elastic_model.computeGradientHessian(q_new)
    R_history[step] = gradE

    # 3) strains & stresses
    for i, edge in enumerate(connectivity):
        _, n0, n1 = edge
        p0 = q_new[3*n0 : 3*n0+3]
        p1 = q_new[3*n1 : 3*n1+3]
        eps = get_strain_stretch_edge2D3D(p0, p1, L0[i])
        strain_history[step, i] = eps
        stress_history[step, i] = EA * eps


# %% Functions for Euler-Bernoulli beam verification


def fun_reaction_force_RightEnd(X0_4columns, R_history, step=-1, axis=1):
    # Compute the total vertical reaction at the right end of the beam.
   
    coords = X0_4columns[:, axis]
    xmax   = coords.max()
    
    right_nodes = np.where(np.isclose(coords, xmax, atol=1e-8))[0]
    dof_z = right_nodes * 3 + 2

    reaction_sum = np.sum(R_history[step, dof_z])
    return reaction_sum, right_nodes, dof_z


def fun_EBBeam_reaction_force(delta, E, h, X0_4columns, axis_length=1, axis_width=2):
    # Compute Euler–Bernoulli reaction force for a cantilever beam under end deflection.

    # Beam length
    coords_L = X0_4columns[:, axis_length]
    L = coords_L.max() - coords_L.min()
    # Beam width
    coords_b = X0_4columns[:, axis_width]
    b = coords_b.max() - coords_b.min()
    
    I = b * h**3 / 12.0
    # Point-load formula: δ = P L^3 / (3 E I) -> P = 3 E I δ / L^3
    P = 3.0 * E * I * delta / L**3
    return P


def fun_deflection_RightEnd(X0_4columns, Q_history, step=-1, axis=1):
    # Compute the vertical deflections at the right end of the beam.

    # extract the coordinate along the beam axis
    coords = X0_4columns[:, axis]
    xmax   = coords.max()
    
    # find the nodes at the right tip
    right_nodes = np.where(np.isclose(coords, xmax, atol=1e-8))[0]
    # global z‐DOF indices for those nodes
    dof_z = right_nodes * 3 + 2
    
    q_step = Q_history[step]
    X0_flat = X0_4columns[:,1:4].ravel()
    
    # compute deflection = current_z - reference_z
    deflections = q_step[dof_z] - X0_flat[dof_z]
    
    return deflections, right_nodes, dof_z