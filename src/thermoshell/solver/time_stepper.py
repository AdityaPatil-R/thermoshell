import numpy as np
from typing import List, Callable, Tuple, Dict
# Added for sparse matrices and solvers
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Imports from other modules (needed by record_step)
from src.thermoshell.material.unit_laws import get_strain_stretch_edge2D3D # Needed for strain logging
from src.thermoshell.bending_model.geometry import getTheta

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
        # Precompute using sparse diagonal matrix intead of np.zeros
        # O(n) time and memory instead of O(n^2)
        self.M_over_dt2   = sp.diags(self.massVector / (self.dt**2), format='csc')

        # Attach BC manager and elastic-energy model
        self.bc           = boundaryCondition
        self.elasticModel = elasticModel
        self.X0 = np.asarray(X0, float).copy()  # length = ndof
        
    def makeMassMatrix(self):
        """Build diagonal mass matrix M."""
        # Precomputed, so no need for np.fill_diagonal
        pass

    def makeWeight(self):
        """Compute gravity‐induced forces per DOF."""
        self.Fg = np.tile(self.g, self.N) * self.massVector

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
        q_new[self.bc.fixedIndices] = self.X0[self.bc.fixedIndices] + self.bc.fixedDOFs

        # 2) Newton‐Raphson loop
        error = np.inf
        for iteration in range(1, self.maxIter+1):
            # internal: gradient & Hessian
            gradE, hessE = self.elasticModel.computeGradientHessian(q_new)
            # inertia: M*((q_new - q_old)/dt - u_old)/dt
            inertiaF = (M/dt) * (((q_new - q_old)/dt) - u_old)
            # residual & Jacobian
            R = inertiaF + gradE - Fg
            # Use precomputed matrix for jacobian rather than recalculating each time
            # Effectively O(n) instead of O(n^2) due to producing a sparse matrix instead of a dense one
            J = self.M_over_dt2 + hessE
            
            # restrict to freeIndex DOFs
            freeIndex = self.bc.freeIndices
            Rf   = R[freeIndex]
            Jf   = J[np.ix_(freeIndex, freeIndex)]
            # Sparse solver instead of dense
            # O(n^1.5) or O(n^2) instead of O(n^3)
            dqf  = spsolve(Jf, Rf)
            
            # update free DOFs
            q_new[freeIndex] -= dqf
            
            error = np.linalg.norm(dqf)
            # print(f"  Newton iter={iteration}: error={error:.4e}")
            if error < self.qtol:
                break
            
        # new velocities and accelerations
        u_new = (q_new - q_old)/dt
        a_new = R/M
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


        # Attach BC manager and elastic-energy model
        self.bc           = boundaryCondition
        self.elasticModel = elasticModel
        self.X0 = np.asarray(X0, float).copy()  # length = ndof
        self.last_num_iters = 0
        
    def makeMassMatrix(self):
        """Build diagonal mass matrix M."""
        pass

    def makeWeight(self):
        """Compute gravity‐induced forces per DOF."""
        self.Fg = np.tile(self.g, self.N) * self.massVector

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
        Fg = self.Fg
        X0 = self.X0

        # 1) Initialize and impose Dirichlet BCs
        q_new = q_guess.copy()
        q_new[self.bc.fixedIndices] = self.X0[self.bc.fixedIndices] + self.bc.fixedDOFs

        # 2) Newton‐Raphson loop
        rel_error = np.inf
        for iteration in range(1, self.maxIter+1):
            # internal: gradient & Hessian
            gradE, hessE = self.elasticModel.computeGradientHessian(q_new)
            # residual & Jacobian
            R = gradE - Fg
            J = hessE
            
            # restrict to freeIndex DOFs
            freeIndex = self.bc.freeIndices
            Rf   = R[freeIndex]
            Jf   = J[np.ix_(freeIndex, freeIndex)]
            # Sparse solver instead of dense
            # O(n^1.5) or O(n^2) instead of O(n^3)
            dqf  = spsolve(Jf, Rf)
            
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


        # Attach BC manager and elastic-energy model
        self.bc           = boundaryCondition
        self.elasticModel = elasticModel
        self.X0 = np.asarray(X0, float).copy()  # length = ndof
        
    def makeMassMatrix(self):
        """Build diagonal mass matrix M."""
        pass

    def makeWeight(self):
        """Compute gravity‐induced forces per DOF."""
        self.Fg = np.tile(self.g, self.N) * self.massVector

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
        Fg = self.Fg
        X0 = self.X0

        # 1) Initialize and impose Dirichlet BCs
        q_new = q_guess.copy()
        q_new[self.bc.fixedIndices] = self.X0[self.bc.fixedIndices] + self.bc.fixedDOFs
            
        # 2) Newton‐Raphson loop
        error = np.inf
        for iteration in range(1, self.maxIter+1):
            # internal: gradient & Hessian
            gradE, hessE = self.elasticModel.computeGradientHessian(q_new)
            # residual & Jacobian
            R = gradE - Fg
            J = hessE
            
            # restrict to freeIndex DOFs
            freeIndex = self.bc.freeIndices
            Rf   = R[freeIndex]
            Jf   = J[np.ix_(freeIndex, freeIndex)]
            # Sparse solver instead of dense
            # O(n^1.5) or O(n^2) instead of O(n^3)
            dqf  = spsolve(Jf, Rf)
            
            # update free DOFs
            q_new[freeIndex] -= dqf
            
            error = np.linalg.norm(dqf)
            if error < self.qtol:
                break
        # new velocities and accelerations
        # u_new = (q_new - q_old)/dt
        # a_new = (inertiaF + gradE - Fg)/M
        # return q_new, u_new, a_new, (error < self.qtol)
        return q_new, (error < self.qtol)
    

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