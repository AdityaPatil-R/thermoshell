import numpy as np

from src.thermoshell.analysis.material.unit_laws import (fun_grad_hess_energy_stretch_linear_elastic_edge, 
                                         fun_grad_hess_energy_stretch_linear_elastic_edge_thermal,
                                         get_strain_stretch_edge2D3D, 
                                         grad_and_hess_strain_stretch_edge3D)

from src.thermoshell.analysis.bending_model.geometry import (getTheta, 
                                             gradTheta, 
                                             hessTheta)

from src.thermoshell.analysis.bending_model.energy import (calculate_pure_bending_grad_hess, 
                                           calculate_coupled_bending_grad_hess, 
                                           calculate_coupled_bending_grad_hess_thermal)

class ElasticGHBase:
    """
    Base class containing common properties and the computeGradientHessian structure.
    """

    def __init__(self,
                 energy_choice: int,
                 Nedges: int,
                 NP_total: int,
                 Ndofs: int,
                 connectivity: np.ndarray,
                 l0_ref: np.ndarray,
                 hinge_quads: np.ndarray,
                 theta_bar: float,
                 model_choice: int=1,
                 nn_model=None):
        
        # Common initialization
        self.connectivity = connectivity.astype(int)
        self.l_ref        = l0_ref
        self.energy_choice= energy_choice
        self.model_choice = model_choice
        self.nn_model     = nn_model
        self.num_edges    = Nedges
        self.num_nodes    = NP_total
        self.ndof         = Ndofs
        self.hinge_quads  = hinge_quads
        self.theta_bar    = float(theta_bar)

    def _assemble_stretch(self, 
                          G: np.ndarray, 
                          H: np.ndarray, 
                          q: np.ndarray):
        """Placeholder, must be overridden by all subclasses."""

        raise NotImplementedError("Subclass must implement _assemble_stretch.")

    def _assemble_bending(self, 
                          G: np.ndarray, 
                          H: np.ndarray, 
                          q: np.ndarray):
        """Placeholder, must be overridden by all subclasses."""

        raise NotImplementedError("Subclass must implement _assemble_bending.")

    def computeGradientHessian(self, 
                               q: np.ndarray):
        """
        Global gradient G and Hessian H from both stretch and bending.
        energy_choice: 1=stretch, 2=bending, 3=both. 
        Note: Subclasses will override this method for energy_choice=4.
        """

        G = np.zeros(self.ndof)
        H = np.zeros((self.ndof, self.ndof))

        if self.energy_choice in (1, 3):
            self._assemble_stretch(G, H, q)

        if self.energy_choice in (2, 3):
            self._assemble_bending(G, H, q)

        return G, H


class ElasticGHCoupledBase(ElasticGHBase):
    """
    Intermediate base class for coupled models:
    - Adds common initialization for array stiffnesses and beta.
    - Adds the common helper _edge_length.
    - _assemble_stretch and _assemble_bending remain NotImplemented here,
      as they are still different between Coupled and CoupledThermal.
    """

    def __init__(self, ks_array, kb_array, Nedges, hinge_quads, **kwargs):
        # Pass common kwargs up to the primary base class
        super_kwargs = {
            'Nedges': Nedges, 
            'hinge_quads': hinge_quads, 
            'ks_array': ks_array, # These will be ignored by ElasticGHBase but are needed locally
            'kb_array': kb_array,
            **kwargs
        }
        
        # Call the parent's constructor using the necessary arguments
        super().__init__(
            energy_choice=super_kwargs.pop('energy_choice'),
            Nedges=super_kwargs.pop('Nedges'),
            NP_total=super_kwargs.pop('NP_total'),
            Ndofs=super_kwargs.pop('Ndofs'),
            connectivity=super_kwargs.pop('connectivity'),
            l0_ref=super_kwargs.pop('l0_ref'),
            hinge_quads=super_kwargs.pop('hinge_quads'),
            theta_bar=super_kwargs.pop('theta_bar'),
            model_choice=super_kwargs.pop('model_choice', 1), # Default from Base
            nn_model=super_kwargs.pop('nn_model', None)      # Default from Base
        )

        ks_array = np.asarray(ks_array, float)

        if ks_array.ndim == 0:
            ks_array = np.full(Nedges, ks_array)
        elif ks_array.shape[0] != Nedges:
            raise ValueError("ks_array must be length Nedges")
        
        self.ks_array = ks_array
        kb_array = np.asarray(kb_array, float)
        n_hinges = 0 if hinge_quads is None else hinge_quads.shape[0]

        if kb_array.ndim == 0:
            kb_array = np.full(n_hinges, kb_array)
        elif kb_array.shape[0] != n_hinges:
            raise ValueError(f"kb_array must have length {n_hinges}")
        
        self.kb_array = kb_array

    def _edge_length(self, a:int, b:int) -> float:
        """Helper: look up reference length between global nodes a,b."""
        
        for eid,n0,n1 in self.connectivity:
            if {n0,n1} == {a,b}:
                return self.l_ref[eid]
            
        raise KeyError(f"edge {a}-{b} not found")

class ElasticGHEdges(ElasticGHBase):
    def __init__(self,
                 energy_choice: int,
                 Nedges: int,
                 NP_total: int,
                 Ndofs: int,
                 connectivity: np.ndarray,
                 l0_ref: np.ndarray,
                 ks: float,
                 hinge_quads: np.ndarray = None,
                 theta_bar: float = 0.0,
                 kb: float = 1.0,
                 h: float = 0.1,
                 model_choice: int=1,
                 nn_model=None):
        
        super().__init__(energy_choice=energy_choice, 
                         Nedges=Nedges, 
                         NP_total=NP_total, 
                         Ndofs=Ndofs, 
                         connectivity=connectivity, 
                         l0_ref=l0_ref, 
                         hinge_quads=hinge_quads, 
                         theta_bar=theta_bar, 
                         model_choice=model_choice, 
                         nn_model=nn_model)
        
        self.ks = ks
        self.kb = kb
        self.h  = h
        
    def _assemble_stretch(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        for eid, n0, n1 in self.connectivity:
            l0 = self.l_ref[eid]

            if self.model_choice == 1:
                dG_s, dH_s = fun_grad_hess_energy_stretch_linear_elastic_edge(q[3*n0:3*n0+3], 
                                                                              q[3*n1:3*n1+3], 
                                                                              l0, 
                                                                              self.ks)
            else:
                raise ValueError("NN model to be defined!")
            
            ind = [3*n0+i for i in (0, 1, 2)] + [3*n1+i for i in (0, 1, 2)]
            G[ind]              += dG_s
            H[np.ix_(ind, ind)] += dH_s

    def _assemble_bending(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        if self.hinge_quads is None:
            raise ValueError("hinge_quads must be provided for bending")
        
        for eid, n0, n1, oppA, oppB in self.hinge_quads:
            nodes = [n0, n1, oppA, oppB]
            ind = np.concatenate([[3*n + i for i in range(3)] for n in nodes])
            x_local = q[ind]
            dG_b, dH_b = calculate_pure_bending_grad_hess(x0=x_local, 
                                             theta_bar=self.theta_bar, 
                                             kb=self.kb)
            G[ind]              += dG_b
            H[np.ix_(ind, ind)] += dH_b
            
class ElasticGHEdgesCoupled(ElasticGHCoupledBase): 
    def __init__(self,
                 energy_choice: int,
                 Nedges: int,
                 NP_total: int,
                 Ndofs: int,
                 connectivity: np.ndarray,
                 l0_ref: np.ndarray,
                 ks_array: np.ndarray,
                 hinge_quads: np.ndarray,
                 theta_bar: float,
                 kb_array: np.ndarray,
                 h: float = 0.1,
                 beta: float = 0.0,
                 model_choice: int=1,
                 nn_model=None):
        
        # Initialize common properties in ElasticGHBase and array stiffnesses in ElasticGHCoupledBase
        super().__init__(ks_array=ks_array, 
                         kb_array=kb_array, 
                         Nedges=Nedges, 
                         hinge_quads=hinge_quads, 
                         energy_choice=energy_choice, 
                         NP_total=NP_total, 
                         Ndofs=Ndofs, 
                         connectivity=connectivity, 
                         l0_ref=l0_ref, 
                         theta_bar=theta_bar, 
                         model_choice=model_choice, 
                         nn_model=nn_model)
        
        self.h = h
        self.beta = beta
        
    def _assemble_stretch(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        for eid, n0, n1 in self.connectivity:
            l0 = self.l_ref[eid]
            ke = self.ks_array[eid] 

            if self.model_choice == 1:
                dG_s, dH_s = fun_grad_hess_energy_stretch_linear_elastic_edge(q[3*n0:3*n0+3], 
                                                                              q[3*n1:3*n1+3], 
                                                                              l0, 
                                                                              ke)
            else:
                raise ValueError("NN model to be defined!")
            
            ind = [3*n0+i for i in (0, 1, 2)] + [3*n1+i for i in (0, 1, 2)]
            G[ind]              += dG_s
            H[np.ix_(ind, ind)] += dH_s

    def _assemble_bending(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        if self.hinge_quads is None:
            raise ValueError("hinge_quads must be provided for bending")
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            nodes = [n0, n1, oppA, oppB]
            ind = np.concatenate([[3*n+i for i in range(3)] for n in nodes])
            x_local = q[ind]
            kb_i = self.kb_array[h_idx]
            dG_b, dH_b = calculate_pure_bending_grad_hess(x0=x_local, 
                                             theta_bar=self.theta_bar, 
                                             kb=kb_i)
            G[ind]              += dG_b
            H[np.ix_(ind, ind)] += dH_b
            
    def _assemble_bending_coupled(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        beta = self.beta
        kb_array = self.kb_array

        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i = kb_array[h_idx]
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n+i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]
            
            # Compute θ, ∇θ, ∇²θ
            θ   = getTheta(xloc)
            gθ  = gradTheta(xloc)
            Hθ  = hessTheta(xloc)
            
            # Build Δε, GDeps, HDeps
            Deps  = 0.0
            GDeps = np.zeros(12)
            HDeps = np.zeros((12,12))
            
            pairs = [(n0,oppA),(n0,oppB),(n1,oppA),(n1,oppB)]
            signs = [       +1,       -1,       +1,       -1]
            
            for (a, b), s in zip(pairs,signs):
                ia = nodes.index(a)
                ib = nodes.index(b)
                loc0 = slice(3*ia, 3*ia+3)
                loc1 = slice(3*ib, 3*ib+3)
                
                x0 = xloc[loc0]
                x1 = xloc[loc1]
                L0 = self._edge_length(a,b)
                
                eps_ab = get_strain_stretch_edge2D3D(x0, x1, L0)
                Deps  += s * eps_ab
                dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)
                
                GDeps[loc0] += s * dG_e[0:3]
                GDeps[loc1] += s * dG_e[3:6]
                
                loc = list(range(3*ia,3*ia+3)) + list(range(3*ib,3*ib+3))
                HDeps[np.ix_(loc,loc)] += s * dH_e
            
            # Form coupled force‐ and stiffness‐like pieces
            f_h = θ - beta * Deps
            C_h = gθ  - beta * GDeps
            
            G[inds] += kb_i * f_h * C_h
            Hloc = kb_i * ( np.outer(C_h,C_h) + f_h * (Hθ - beta * HDeps) )
            H[np.ix_(inds,inds)] += Hloc

    def _assemble_bending_coupled_v2(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i  = self.kb_array[h_idx]
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n + i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]
            dG_bh, dH_bh = calculate_coupled_bending_grad_hess(xloc, 
                                                       nodes, 
                                                       self._edge_length, 
                                                       self.beta, 
                                                       kb_i)
            G[inds]               += dG_bh
            H[np.ix_(inds, inds)] += dH_bh
            
    def computeGradientHessian(self, q: np.ndarray):
        G = np.zeros(self.ndof)
        H = np.zeros((self.ndof,self.ndof))
        
        if self.energy_choice in (1,3,4):
            self._assemble_stretch(G,H,q)
        
        if self.energy_choice in (2, 3):
            self._assemble_bending(G,H,q)
        
        if self.energy_choice == 4:
            self._assemble_bending_coupled_v2(G,H,q) 
        
        return G, H

class ElasticGHEdgesCoupledThermal(ElasticGHCoupledBase):
    def __init__(self,
                 energy_choice: int,
                 Nedges: int,
                 NP_total: int,
                 Ndofs: int,
                 connectivity: np.ndarray,
                 l0_ref: np.ndarray,
                 ks_array: np.ndarray,
                 hinge_quads: np.ndarray,
                 theta_bar: float,
                 kb_array: np.ndarray,
                 beta: float,
                 epsilon_th: np.ndarray,
                 model_choice: int=1,
                 nn_model=None):
        
        super().__init__(ks_array=ks_array, 
                         kb_array=kb_array, 
                         Nedges=Nedges, 
                         hinge_quads=hinge_quads, 
                         energy_choice=energy_choice, 
                         NP_total=NP_total, 
                         Ndofs=Ndofs, 
                         connectivity=connectivity, 
                         l0_ref=l0_ref, 
                         theta_bar=theta_bar, 
                         model_choice=model_choice, 
                         nn_model=nn_model)
        
        self.beta = beta
        self.eps_th = epsilon_th
        self.edge_dict = {tuple(sorted((i, j))): eid for eid, i, j in self.connectivity}
        
    def _assemble_stretch(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        for eid, n0, n1 in self.connectivity:
            l0 = self.l_ref[eid]
            ke = self.ks_array[eid]

            if self.model_choice == 1:
                dG_s, dH_s = fun_grad_hess_energy_stretch_linear_elastic_edge_thermal(q[3*n0:3*n0+3], 
                                                                                      q[3*n1:3*n1+3], 
                                                                                      l0, 
                                                                                      ke, 
                                                                                      eps_th=self.eps_th[eid])
            else:
                raise ValueError("NN model to be defined!")
            
            ind = [3*n0+i for i in (0, 1, 2)] + [3*n1+i for i in (0, 1, 2)]
            G[ind]              += dG_s
            H[np.ix_(ind, ind)] += dH_s

    def _assemble_bending(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        if self.hinge_quads is None:
            raise ValueError("hinge_quads must be provided for bending")
        
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i    = self.kb_array[h_idx]
            nodes   = [n0, n1, oppA, oppB]
            ind     = np.concatenate([[3*n + i for i in range(3)] for n in nodes])
            x_local = q[ind]
            dG_b, dH_b = calculate_pure_bending_grad_hess(x0=x_local, 
                                             theta_bar=self.theta_bar, 
                                             kb=kb_i)
            G[ind]              += dG_b
            H[np.ix_(ind, ind)] += dH_b

    def _assemble_bending_coupled(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        beta = self.beta
        kb_array = self.kb_array

        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i  = kb_array[h_idx]
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n+i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]
            
            θ  = getTheta(xloc)
            gθ = gradTheta(xloc)
            Hθ = hessTheta(xloc)
            
            Deps  = 0.0
            GDeps = np.zeros(12)
            HDeps = np.zeros((12,12))
            
            pairs = [(n0,oppA),(n0,oppB),(n1,oppA),(n1,oppB)]
            signs = [       +1,       -1,       +1,       -1]
            
            for (a,b), s in zip(pairs,signs):
                ia = nodes.index(a)
                ib = nodes.index(b)
                loc0 = slice(3*ia, 3*ia+3)
                loc1 = slice(3*ib, 3*ib+3)
                
                x0 = xloc[loc0]
                x1 = xloc[loc1]
                L0 = self._edge_length(a,b)
                
                eps_ab     = get_strain_stretch_edge2D3D(x0, x1, L0)
                Deps      += s * eps_ab
                dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)
                
                GDeps[loc0] += s * dG_e[0:3]
                GDeps[loc1] += s * dG_e[3:6]
                
                loc = list(range(3*ia, 3*ia+3)) + list(range(3*ib, 3*ib+3))
                HDeps[np.ix_(loc,loc)] += s * dH_e
            
            f_h = θ  - beta * Deps
            C_h = gθ - beta * GDeps
            
            G[inds] += kb_i * f_h * C_h
            Hloc = kb_i * (np.outer(C_h,C_h) + f_h * (Hθ - beta * HDeps))
            H[np.ix_(inds,inds)] += Hloc

    def _assemble_bending_coupled_v2(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i  = self.kb_array[h_idx] 
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n+i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]

            dG_bh, dH_bh = calculate_coupled_bending_grad_hess(xloc, 
                                                       nodes, 
                                                       self._edge_length, 
                                                       self.beta, 
                                                       kb_i)
            
            G[inds]               += dG_bh
            H[np.ix_(inds, inds)] += dH_bh
        
        
    def _assemble_bending_coupled_v3(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i  = self.kb_array[h_idx] 
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n + i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]

            dG_bh, dH_bh = calculate_coupled_bending_grad_hess_thermal(xloc, 
                                                               nodes, 
                                                               self._edge_length, 
                                                               self.beta, 
                                                               kb_i,self.eps_th, 
                                                               self.edge_dict)

            G[inds]               += dG_bh
            H[np.ix_(inds, inds)] += dH_bh
    
    def computeGradientHessian(self, q: np.ndarray):
        G = np.zeros(self.ndof)
        H = np.zeros((self.ndof,self.ndof))
        
        if self.energy_choice in (1,3,4):
            self._assemble_stretch(G,H,q)
        
        if self.energy_choice in (2, 3):
            self._assemble_bending(G,H,q)
        
        if self.energy_choice == 4:
            self._assemble_bending_coupled_v3(G,H,q) 
        
        return G, H
    
    # Need to verify the assembly. Done
    # Then modify FDM_ElasticForce to verify. Done

def FDM_ElasticGHEdges(energy_choice,Nedges,NP_total,Ndofs,connectivity,l0_ref, 
                       X0, iPrint, HingeQuads_order, theta_bar, plt):
    
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
    eps:float=1e-6,
    plt=None
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
    eps:float=1e-6,
    plt=None
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
        model_choice=model_choice,
        plt=None
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

