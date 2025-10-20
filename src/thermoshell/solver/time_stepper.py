import numpy as np
from typing import List, Callable, Tuple, Dict

# Imports from other modules (needed by record_step)
from analysis.material.unit_laws import get_strain_stretch_edge2D3D # Needed for strain logging
from analysis.bending_model.geometry import getTheta

class BaseTimeStepper3D:
    """
    Base class for implicit Newton time-steppers for 3D DOFs.
    Contains common initialization and helper methods.
    """

    def __init__(self, massVector, dt, qtol, maxIter, g, boundaryCondition, elasticModel, X0):
        self.massVector = np.asarray(massVector, float)
        self.ndof       = len(self.massVector)

        if self.ndof % 3 != 0:
            raise ValueError("ndof must be a multiple of 3 for 3 DOFs/node")
        
        self.N       = self.ndof // 3
        self.dt      = float(dt)
        self.qtol    = float(qtol)
        self.maxIter = int(maxIter)
        self.g       = np.asarray(g, float)

        if self.g.shape != (3,):
            raise ValueError("Gravity vector must have length 3")

        # Precompute weight Fg = M * g per node
        self.Fg = np.zeros(self.ndof)
        self.makeWeight() # Vectorized

        # Attach BC manager and elastic-energy model
        self.bc           = boundaryCondition
        self.elasticModel = elasticModel
        self.X0           = np.asarray(X0, float).copy()  # length = ndof

    def makeWeight(self):
        """(Vectorized) Compute gravity-induced forces per DOF."""

        # Reshape mass vector to (N, 3), multiply by (3,) gravity vector
        # (which broadcasts), and reshape back to (ndof,)
        self.Fg = (self.massVector.reshape(self.N, 3) * self.g).ravel()

    def _initialize_q_new(self, q_guess: np.ndarray) -> np.ndarray:
        """(Vectorized) Initialize q_new and impose Dirichlet BCs."""

        q_new = q_guess.copy()

        # Convert BC lists to NumPy arrays for vectorized operations
        fixed_indices = np.asarray(self.bc.fixedIndices)
        fixed_dofs = np.asarray(self.bc.fixedDOFs)

        # Vectorized application of boundary conditions
        if fixed_indices.size > 0:
            q_new[fixed_indices] = self.X0[fixed_indices] + fixed_dofs

        return q_new

    def beforeTimeStep(self) -> None:
        pass 

    def afterTimeStep(self) -> None:
        pass 

    def simulate(self, q_guess, q_old, u_old, a_old):
        """Abstract simulate method. Child classes must override this."""
        raise NotImplementedError("Subclass must implement abstract method")

class timeStepper3D(BaseTimeStepper3D):
    """
    Dynamic implicit Newton time-stepper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def simulate(self, q_guess, q_old, u_old, a_old):
        """
        One implicit Newton step (Dynamic).
        """

        dt        = self.dt
        M_vec     = self.massVector # 1D Mass vector
        Fg        = self.Fg
        freeIndex = self.bc.freeIndices

        # Initialize and impose Dirichlet BCs (Vectorized)
        q_new = self._initialize_q_new(q_guess)

        # Newton-Raphson loop
        error = np.inf

        for iteration in range(1, self.maxIter + 1):
            gradE, hessE = self.elasticModel.computeGradientHessian(q_new)
            
            inertiaF = (M_vec / dt) * (((q_new - q_old) / dt) - u_old)
            
            # Residual
            R = inertiaF + gradE - Fg
            
            # Avoid forming a full (ndof, ndof) mass matrix.
            # Add the diagonal term directly to the Hessian.
            J = hessE.copy()
            J.flat[::self.ndof + 1] += M_vec / (dt**2)
            
            # Restrict to freeIndex DOFs
            Rf = R[freeIndex]
            Jf = J[np.ix_(freeIndex, freeIndex)]
            
            try:
                dqf  = np.linalg.solve(Jf, Rf)
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in Newton step. Simulation may fail.")
                return q_new, u_old, a_old, False # Return previous state
            
            # Update free DOFs
            q_new[freeIndex] -= dqf
            error = np.linalg.norm(dqf)

            if error < self.qtol:
                break
        
        # New velocities and accelerations
        u_new = (q_new - q_old) / dt
        a_new = (inertiaF + gradE - Fg) / M_vec 

        return q_new, u_new, a_new, (error < self.qtol)

class timeStepper3D_static(BaseTimeStepper3D):
    """
    Static implicit Newton solver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_num_iters = 0

    def simulate(self, q_guess, q_old, u_old, a_old):
        """
        One implicit Newton step (Static).
        Ignores dynamic terms (q_old, u_old, a_old, dt).
        """

        Fg        = self.Fg
        freeIndex = self.bc.freeIndices

        # Initialize and impose Dirichlet BCs (Vectorized)
        q_new = self._initialize_q_new(q_guess)

        # Newton-Raphson loop
        rel_error = np.inf
        for iteration in range(1, self.maxIter + 1):
            gradE, hessE = self.elasticModel.computeGradientHessian(q_new)
            
            # Residual & Jacobian (Static: R = F_internal - F_external)
            R = gradE - Fg
            J = hessE
            
            # Restrict to freeIndex DOFs
            Rf = R[freeIndex]
            Jf = J[np.ix_(freeIndex, freeIndex)]
            
            try:
                dqf  = np.linalg.solve(Jf, Rf)
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in Newton step. Simulation may fail.")
                self.last_num_iters = iteration
                return q_new, False
            
            qfree     = q_new[freeIndex]
            rel_error = np.linalg.norm(dqf) / max(0.1, np.linalg.norm(qfree))
            
            # Update free DOFs
            q_new[freeIndex] -= dqf
            
            if rel_error < self.qtol:
                self.last_num_iters = iteration
                break
        else: 
            self.last_num_iters = self.maxIter
            
        return q_new, (rel_error < self.qtol)

class timeStepper3D_static_gravity(BaseTimeStepper3D):
    """
    Static solver, but its simulate signature matches the dynamic one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def simulate(self, q_guess, q_old, u_old, a_old):
        """
        One implicit Newton step (Static).
        Ignores dynamic terms (q_old, u_old, a_old, dt).
        """
        Fg = self.Fg
        freeIndex = self.bc.freeIndices

        # Initialize and impose Dirichlet BCs (Vectorized)
        q_new = self._initialize_q_new(q_guess)

        # Newton-Raphson loop
        error = np.inf
        for iteration in range(1, self.maxIter + 1):
            gradE, hessE = self.elasticModel.computeGradientHessian(q_new)

            # Residual & Jacobian (Static)
            R = gradE - Fg
            J = hessE
            
            # restrict to freeIndex DOFs
            Rf = R[freeIndex]
            Jf = J[np.ix_(freeIndex, freeIndex)]
            
            try:
                dqf = np.linalg.solve(Jf, Rf)
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in Newton step. Simulation may fail.")
                return q_new, False
            
            # Update free DOFs
            q_new[freeIndex] -= dqf
            error = np.linalg.norm(dqf)

            if error < self.qtol:
                break
            
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
    Record simulation state at time-step `step`.
    """
    # Record nodal displacements
    Q_history[step] = q_new

    # Record reaction = gradient of total energy
    gradE, _        = elastic_model.computeGradientHessian(q_new)
    R_history[step] = gradE

    # Record edge strains & stresses
    n0s = connectivity[:, 1]
    n1s = connectivity[:, 2]
    q_r = q_new.reshape(-1, 3)
    p0s = q_r[n0s] # (Nedges, 3)
    p1s = q_r[n1s] # (Nedges, 3)
    
    # Vectorized computation of all current lengths
    L_current_all = np.linalg.norm(p1s - p0s, axis=1)
    length_history[step] = L_current_all
    
    for i in range(connectivity.shape[0]):
        eps = get_strain_stretch_edge2D3D(p0s[i], p1s[i], L0[i])

        strain_history[step, i] = eps
        stress_history[step, i] = ks_array[i] * eps

    # Record dihedral angle at each hinge
    for j, (_, n0, n1, oppA, oppB) in enumerate(hinge_quads):
        x0 = q_new[3*n0   : 3*n0+3]
        x1 = q_new[3*n1   : 3*n1+3]
        x2 = q_new[3*oppA : 3*oppA+3]
        x3 = q_new[3*oppB : 3*oppB+3]
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
    Record simulation state at time-step `step`.
    """
    # Displacement
    Q_history[step] = q_new

    # Reaction = gradient of elastic energy
    gradE, _        = elastic_model.computeGradientHessian(q_new)
    R_history[step] = gradE

    # Vectorized "gather" of node positions
    n0s = connectivity[:, 1]
    n1s = connectivity[:, 2]
    q_r = q_new.reshape(-1, 3)
    p0s = q_r[n0s] # (Nedges, 3)
    p1s = q_r[n1s] # (Nedges, 3)

    for i in range(connectivity.shape[0]):
        eps = get_strain_stretch_edge2D3D(p0s[i], p1s[i], L0[i])
        strain_history[step, i] = eps
        stress_history[step, i] = EA * eps