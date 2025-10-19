import numpy as np

# From elasticity.py
from elasticity import (
    fun_grad_hess_energy_stretch_linear_elastic_edge, 
    fun_grad_hess_energy_stretch_linear_elastic_edge_thermal,
    get_strain_stretch_edge2D3D, 
    grad_and_hess_strain_stretch_edge3D
)

from analysis.bending_model.geometry import (
    getTheta, 
    gradTheta, 
    hessTheta
)

# Differential Stretch (from analysis/bending_model/stretch_diff.py)
from analysis.bending_model.stretch_diff import (
    fun_DEps_grad_hess, 
    fun_DEps_grad_hess_thermal
)

# Energy Models (from analysis/bending_model/energy.py)
from analysis.bending_model.energy import (
    gradEb_hessEb_Shell, 
    fun_coupled_Ebend_grad_hess, 
    fun_coupled_Ebend_grad_hess_thermal
)

class ElasticGHEdges:
    def __init__(self,
                 energy_choice: int,  # 1: stretch only, 2: bending only, 3: both
                 Nedges: int,
                 NP_total: int,
                 Ndofs: int,
                 connectivity: np.ndarray,   # shape (Nedges,2)
                 l0_ref: np.ndarray,         # shape (Nedges,)
                 ks: float,
                 hinge_quads: np.ndarray = None,
                 theta_bar: float = 0.0,
                 kb: float = 1.0,
                 h: float = 0.1,
                 model_choice: int=1,
                 nn_model=None):
        
        self.connectivity = connectivity.astype(int)
        self.l_ref        = l0_ref
        self.ks           = ks
        self.kb           = kb
        self.h            = h
        self.energy_choice= energy_choice
        self.model_choice = model_choice
        self.nn_model     = nn_model
        self.num_edges    = Nedges
        self.num_nodes    = NP_total
        self.ndof         = Ndofs
        self.hinge_quads  = hinge_quads
        self.theta_bar    = float(theta_bar)
        
    def _assemble_stretch(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """Add stretch‐energy contributions into G,H."""
        for eid, n0, n1 in self.connectivity:
            l0 = self.l_ref[eid]
            # choose analytic or NN‐based
            if self.model_choice == 1:
                dG_s, dH_s = fun_grad_hess_energy_stretch_linear_elastic_edge(
                    q[3*n0:3*n0+3], q[3*n1:3*n1+3], l0, self.ks)
            else:
                raise ValueError("NN model to be defined!")
                # placeholder for a neural‐net‐based call
                # dG_s, dH_s = self.nn_model.predict_stretch(q, eid)

            ind = [3*n0 + i for i in (0,1,2)] + [3*n1 + i for i in (0,1,2)]
            G[ind]              += dG_s
            H[np.ix_(ind, ind)] += dH_s

    def _assemble_bending(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """Add bending‐energy contributions into G,H."""
        if self.hinge_quads is None:
            raise ValueError("hinge_quads must be provided for bending")
        for eid, n0, n1, oppA, oppB in self.hinge_quads:
            nodes = [n0, n1, oppA, oppB]
            # build the 12‐length local DOF vector
            ind = np.concatenate([[3*n + i for i in range(3)] for n in nodes])
            x_local = q[ind]

            # analytic bending gradient & Hessian
            dG_b, dH_b = gradEb_hessEb_Shell(
                x0=x_local,
                theta_bar=self.theta_bar,
                kb=self.kb)

            G[ind]              += dG_b
            H[np.ix_(ind, ind)] += dH_b

    def computeGradientHessian(self, q: np.ndarray):
        """
        Global gradient G and Hessian H from both stretch and bending.
        energy_choice: 1=stretch, 2=bending, 3=both
        """
        G = np.zeros(self.ndof)
        H = np.zeros((self.ndof, self.ndof))

        if self.energy_choice in (1, 3):
            self._assemble_stretch(G, H, q)

        if self.energy_choice in (2, 3):
            self._assemble_bending(G, H, q)

        return G, H

    

class ElasticGHEdgesCoupledThermal:
    def __init__(self,
                 energy_choice: int,  # 1: stretch only, 2: bending only, 3: both
                 Nedges: int,
                 NP_total: int,
                 Ndofs: int,
                 connectivity: np.ndarray,   # shape (Nedges,2)
                 l0_ref: np.ndarray,         # shape (Nedges,)
                 ks_array: np.ndarray,
                 hinge_quads: np.ndarray,
                 theta_bar: float,
                 kb_array: np.ndarray,
                 beta: float,
                 epsilon_th: np.ndarray,
                 model_choice: int=1,
                 nn_model=None):
        
        self.connectivity = connectivity.astype(int)
        self.l_ref        = l0_ref
        # self.ks           = ks
        # self.kb           = kb
        # self.h            = h
        self.beta         = beta
        self.energy_choice= energy_choice
        self.model_choice = model_choice
        self.nn_model     = nn_model
        self.num_edges    = Nedges
        self.num_nodes    = NP_total
        self.ndof         = Ndofs
        self.hinge_quads  = hinge_quads
        self.theta_bar    = float(theta_bar)
        self.eps_th       = epsilon_th
        # sort just the two node‑indices so they always come out (min_node, max_node)
        self.edge_dict = {
            tuple(sorted((i, j))): eid
            for eid, i, j in self.connectivity}
        
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
        
        
    def _assemble_stretch(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """Add stretch‐energy contributions into G,H."""
        for eid, n0, n1 in self.connectivity:
            l0 = self.l_ref[eid]
            ke = self.ks_array[eid]
            # choose analytic or NN‐based
            if self.model_choice == 1:
                dG_s, dH_s = fun_grad_hess_energy_stretch_linear_elastic_edge_thermal(
                    q[3*n0:3*n0+3], q[3*n1:3*n1+3], l0, ke, eps_th=self.eps_th[eid])
                
                # dG_s, dH_s = fun_grad_hess_energy_stretch_linear_elastic_edge(
                #     q[3*n0:3*n0+3], q[3*n1:3*n1+3], l0, self.ks)
            else:
                raise ValueError("NN model to be defined!")
                # placeholder for a neural‐net‐based call
                # dG_s, dH_s = self.nn_model.predict_stretch(q, eid)

            ind = [3*n0 + i for i in (0,1,2)] + [3*n1 + i for i in (0,1,2)]
            G[ind]              += dG_s
            H[np.ix_(ind, ind)] += dH_s

    def _assemble_bending(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """Add bending‐energy contributions into G,H."""
        if self.hinge_quads is None:
            raise ValueError("hinge_quads must be provided for bending")
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i = self.kb_array[h_idx]
            nodes = [n0, n1, oppA, oppB]
            # build the 12‐length local DOF vector
            ind = np.concatenate([[3*n + i for i in range(3)] for n in nodes])
            x_local = q[ind]

            # analytic bending gradient & Hessian
            dG_b, dH_b = gradEb_hessEb_Shell(
                x0=x_local,
                theta_bar=self.theta_bar,
                kb=kb_i)

            G[ind]              += dG_b
            H[np.ix_(ind, ind)] += dH_b
            
    def _edge_length(self, a:int, b:int) -> float:
        """Helper: look up reference length between global nodes a,b."""
        # find the edge ID for (a,b) in self.connectivity
        # you can build a dict in __init__ for speed; here’s the simplest:
        for eid,n0,n1 in self.connectivity:
            if {n0,n1} == {a,b}:
                return self.l_ref[eid]
        raise KeyError(f"edge {a}-{b} not found")


    def _assemble_bending_coupled(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """
        Coupled bending: Σ_h ½·kb·[θ_h − β·Δε_h]²,
        with Δε_h = ε(n0–oppA) − ε(n1–oppA) + ε(n1–oppB) − ε(n0–oppB).
        """
        beta = self.beta
        kb_array = self.kb_array
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i  = kb_array[h_idx]
            # global indices for this quad’s 4 nodes
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n+i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]   # length 12
            
            # 1) compute θ, ∇θ, ∇²θ
            θ   = getTheta(xloc)
            gθ  = gradTheta(xloc)
            Hθ  = hessTheta(xloc)
            
            # 2) build Δε, GDeps, HDeps
            Deps  = 0.0
            GDeps = np.zeros(12)
            HDeps = np.zeros((12,12))
            
            pairs = [(n0,oppA),(n0,oppB),(n1,oppA),(n1,oppB)]
            signs = [+1,      -1,      +1,      -1]
            
            for (a,b), s in zip(pairs,signs):
                ia = nodes.index(a)
                ib = nodes.index(b)
                loc0 = slice(3*ia,   3*ia+3)
                loc1 = slice(3*ib,   3*ib+3)
                
                x0 = xloc[loc0]
                x1 = xloc[loc1]
                L0 = self._edge_length(a,b)
                
                # scalar stretch
                eps_ab = get_strain_stretch_edge2D3D(x0, x1, L0)
                Deps  += s * eps_ab
                
                # its grad & hess (6→[x0,x1])
                dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)
                
                # scatter into our 12‐vector
                GDeps[loc0] += s * dG_e[0:3]
                GDeps[loc1] += s * dG_e[3:6]
                
                loc = list(range(3*ia,3*ia+3)) + list(range(3*ib,3*ib+3))
                HDeps[np.ix_(loc,loc)] += s * dH_e
            
            # 3) form coupled force‐ and stiffness‐like pieces
            f_h = θ - beta * Deps      # scalar
            C_h = gθ  - beta * GDeps   # shape (12,)
            
            # gradient contribution
            G[inds] += kb_i * f_h * C_h
            
            # Hessian contribution
            Hloc = kb_i * ( np.outer(C_h,C_h) + f_h * (Hθ - beta * HDeps) )
            H[np.ix_(inds,inds)] += Hloc


    def _assemble_bending_coupled_v2(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """
        Same as _assemble_bending_coupled but calling fun_coupled_Ebend_grad_hess.
        """
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i = self.kb_array[h_idx] 
            # 1) collect the 4 global node‐IDs and their dof‐indices
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n + i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]   # length‑12 slice

            # 2) delegate to our helper
            dG_bh, dH_bh = fun_coupled_Ebend_grad_hess(
                xloc,
                nodes,
                self._edge_length,
                self.beta,
                kb_i
            )

            # 3) scatter into the global arrays
            G[inds]              += dG_bh
            H[np.ix_(inds, inds)] += dH_bh
        
        
    def _assemble_bending_coupled_v3(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """
        Same as _assemble_bending_coupled_v2 but calling fun_coupled_Ebend_grad_hess_thermal.
        """
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i = self.kb_array[h_idx] 
            # 1) collect the 4 global node‐IDs and their dof‐indices
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n + i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]   # length‑12 slice

            # 2) delegate to our helper
            dG_bh, dH_bh = fun_coupled_Ebend_grad_hess_thermal(
                xloc,
                nodes,
                self._edge_length,
                self.beta,
                kb_i,
                self.eps_th,
                self.edge_dict
            )

            # 3) scatter into the global arrays
            G[inds]              += dG_bh
            H[np.ix_(inds, inds)] += dH_bh
    
    
    def computeGradientHessian(self, q: np.ndarray):
        """
        Global gradient G and Hessian H from both stretch and bending.
        energy_choice: 1=stretch, 2=bending, 3=both, 4 coupled bending
        """
        G = np.zeros(self.ndof)
        H = np.zeros((self.ndof,self.ndof))
        
        if self.energy_choice in (1,3,4):
            self._assemble_stretch(G,H,q)
        
        # uncoupled bending
        if self.energy_choice in (2, 3):
            self._assemble_bending(G,H,q)
        
        # coupled bending
        if self.energy_choice == 4:
            # self._assemble_bending_coupled(G,H,q)
            # self._assemble_bending_coupled_v2(G,H,q)
            self._assemble_bending_coupled_v3(G,H,q)
        
        return G, H
    

class ElasticGHEdgesCoupled: #_BeforeThermal modification 05-10-25 2pm.
    def __init__(self,
                 energy_choice: int,  # 1: stretch only, 2: bending only, 3: both
                 Nedges: int,
                 NP_total: int,
                 Ndofs: int,
                 connectivity: np.ndarray,   # shape (Nedges,2)
                 l0_ref: np.ndarray,         # shape (Nedges,)
                 ks_array: np.ndarray,
                 hinge_quads: np.ndarray,
                 theta_bar: float,
                 kb_array: np.ndarray,
                 h: float = 0.1,
                 beta: float = 0.0,
                 model_choice: int=1,
                 nn_model=None):
        
        self.connectivity = connectivity.astype(int)
        self.l_ref        = l0_ref
        # self.ks           = ks
        # self.kb           = kb
        self.h            = h
        self.beta         = beta
        self.energy_choice= energy_choice
        self.model_choice = model_choice
        self.nn_model     = nn_model
        self.num_edges    = Nedges
        self.num_nodes    = NP_total
        self.ndof         = Ndofs
        self.hinge_quads  = hinge_quads
        self.theta_bar    = float(theta_bar)
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
        
        
    def _assemble_stretch(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """Add stretch‐energy contributions into G,H."""
        for eid, n0, n1 in self.connectivity:
            l0 = self.l_ref[eid]
            ke = self.ks_array[eid] 
            # choose analytic or NN‐based
            if self.model_choice == 1:
                dG_s, dH_s = fun_grad_hess_energy_stretch_linear_elastic_edge(
                    q[3*n0:3*n0+3], q[3*n1:3*n1+3], l0, ke)
            else:
                raise ValueError("NN model to be defined!")
                # placeholder for a neural‐net‐based call
                # dG_s, dH_s = self.nn_model.predict_stretch(q, eid)

            ind = [3*n0 + i for i in (0,1,2)] + [3*n1 + i for i in (0,1,2)]
            G[ind]              += dG_s
            H[np.ix_(ind, ind)] += dH_s

    def _assemble_bending(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """Add bending‐energy contributions into G,H."""
        if self.hinge_quads is None:
            raise ValueError("hinge_quads must be provided for bending")
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            nodes = [n0, n1, oppA, oppB]
            # build the 12‐length local DOF vector
            ind = np.concatenate([[3*n + i for i in range(3)] for n in nodes])
            x_local = q[ind]
            kb_i = self.kb_array[h_idx]         # <-- per-hinge

            # analytic bending gradient & Hessian
            dG_b, dH_b = gradEb_hessEb_Shell(
                x0=x_local,
                theta_bar=self.theta_bar,
                kb=kb_i)

            G[ind]              += dG_b
            H[np.ix_(ind, ind)] += dH_b
            
    def _edge_length(self, a:int, b:int) -> float:
        """Helper: look up reference length between global nodes a,b."""
        # find the edge ID for (a,b) in self.connectivity
        # you can build a dict in __init__ for speed; here’s the simplest:
        for eid,n0,n1 in self.connectivity:
            if {n0,n1} == {a,b}:
                return self.l_ref[eid]
        raise KeyError(f"edge {a}-{b} not found")


    def _assemble_bending_coupled(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """
        Coupled bending: Σ_h ½·kb·[θ_h − β·Δε_h]²,
        with Δε_h = ε(n0–oppA) − ε(n1–oppA) + ε(n1–oppB) − ε(n0–oppB).
        """
        beta = self.beta
        kb_array = self.kb_array
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i = kb_array[h_idx]   # per‑hinge
            # global indices for this quad’s 4 nodes
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n+i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]   # length 12
            
            # 1) compute θ, ∇θ, ∇²θ
            θ   = getTheta(xloc)
            gθ  = gradTheta(xloc)
            Hθ  = hessTheta(xloc)
            
            # 2) build Δε, GDeps, HDeps
            Deps  = 0.0
            GDeps = np.zeros(12)
            HDeps = np.zeros((12,12))
            
            pairs = [(n0,oppA),(n0,oppB),(n1,oppA),(n1,oppB)]
            signs = [+1,      -1,      +1,      -1]
            
            for (a,b), s in zip(pairs,signs):
                ia = nodes.index(a)
                ib = nodes.index(b)
                loc0 = slice(3*ia,   3*ia+3)
                loc1 = slice(3*ib,   3*ib+3)
                
                x0 = xloc[loc0]
                x1 = xloc[loc1]
                L0 = self._edge_length(a,b)
                
                # scalar stretch
                eps_ab = get_strain_stretch_edge2D3D(x0, x1, L0)
                Deps  += s * eps_ab
                
                # its grad & hess (6→[x0,x1])
                dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)
                
                # scatter into our 12‐vector
                GDeps[loc0] += s * dG_e[0:3]
                GDeps[loc1] += s * dG_e[3:6]
                
                loc = list(range(3*ia,3*ia+3)) + list(range(3*ib,3*ib+3))
                HDeps[np.ix_(loc,loc)] += s * dH_e
            
            # 3) form coupled force‐ and stiffness‐like pieces
            f_h = θ - beta * Deps      # scalar
            C_h = gθ  - beta * GDeps   # shape (12,)
            
            # gradient contribution
            G[inds] += kb_i * f_h * C_h
            
            # Hessian contribution
            Hloc = kb_i * ( np.outer(C_h,C_h) + f_h * (Hθ - beta * HDeps) )
            H[np.ix_(inds,inds)] += Hloc


    def _assemble_bending_coupled_v2(self, G: np.ndarray, H: np.ndarray, q: np.ndarray):
        """
        Same as _assemble_bending_coupled but calling fun_coupled_Ebend_grad_hess.
        """
        for h_idx, (eid, n0, n1, oppA, oppB) in enumerate(self.hinge_quads):
            kb_i = self.kb_array[h_idx]
            # 1) collect the 4 global node‐IDs and their dof‐indices
            nodes = [n0, n1, oppA, oppB]
            inds  = sum([[3*n + i for i in range(3)] for n in nodes], [])
            xloc  = q[inds]   # length‑12 slice

            # 2) delegate to our helper
            dG_bh, dH_bh = fun_coupled_Ebend_grad_hess(
                xloc,
                nodes,
                self._edge_length,
                self.beta,
                kb_i
            )

            # 3) scatter into the global arrays
            G[inds]              += dG_bh
            H[np.ix_(inds, inds)] += dH_bh
            
    def computeGradientHessian(self, q: np.ndarray):
        """
        Global gradient G and Hessian H from both stretch and bending.
        energy_choice: 1=stretch, 2=bending, 3=both, 4 coupled bending
        """
        G = np.zeros(self.ndof)
        H = np.zeros((self.ndof,self.ndof))
        
        if self.energy_choice in (1,3,4):
            self._assemble_stretch(G,H,q)
        
        # uncoupled bending
        if self.energy_choice in (2, 3):
            self._assemble_bending(G,H,q)
        
        # coupled bending
        if self.energy_choice == 4:
            # self._assemble_bending_coupled(G,H,q)
            self._assemble_bending_coupled_v2(G,H,q)
        
        return G, H