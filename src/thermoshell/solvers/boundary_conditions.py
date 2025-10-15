# %% Def. a class and functions: def disp BC, fiexed, free DOFs and variation with history.
# understand the class BoundaryConditions, update. Done

class BoundaryConditions3D:
    def __init__(self, ndof):
        """
        Initialize boundary conditions.
            ndof (int): Total number of degrees of freedom (DOFs).
        """
        if ndof % 3 != 0:
            raise ValueError("ndof must be a multiple of 3 for 3-DOF nodes.")
        self.ndof = ndof
        self.fixedIndices = []  # List for fixed DOFs
        self.fixedDOFs = []  # List for fixed boundary condition values
        self.freeIndices = list(range(ndof))  # Initialize free DOFs (0 to ndof-1)
        

    def setBoundaryConditionNode(self, i, x):
        """
        Add or update boundary conditions for a node.
            i (int): Node index.
            x (numpy.ndarray): array [u_x, u_y, u_z] representing boundary condition values.
        """
        x = np.asarray(x, float)
        if x.shape != (3,):
            raise ValueError("x must be array of length 3 for (ux,uy,uz).")

        # Calculate the DOFs for this node
        dof_1, dof_2, dof_3 = 3*i, 3*i+1, 3*i+2

        # Check if DOFs are already in fixedDOFs
        if dof_1 in self.fixedIndices and dof_2 in self.fixedIndices and dof_3 in self.fixedIndices:
            # Find their indices in fixedDOFs
            index_1 = self.fixedIndices.index(dof_1)
            index_2 = self.fixedIndices.index(dof_2)
            index_3 = self.fixedIndices.index(dof_3)

            # Update corresponding values in fixedIndices
            self.fixedDOFs[index_1] = x[0]
            self.fixedDOFs[index_2] = x[1]
            self.fixedDOFs[index_3] = x[2]
        else:
            # Add new DOFs and corresponding values
            self.fixedIndices.extend([dof_1, dof_2, dof_3])
            self.fixedDOFs.extend(x.copy().tolist())  # Convert NumPy array to list for storage

            # Remove the DOFs from freeIndices
            self.freeIndices.remove(dof_1)
            self.freeIndices.remove(dof_2)
            self.freeIndices.remove(dof_3)

    def releaseBoundaryConditionNode(self, i):
        """
        Release the boundary conditions for a node.
            i (int): Node index to release.
        """
        # Calculate the DOFs for this node
        dof_1, dof_2, dof_3 = 3*i, 3*i+1, 3*i+2

        # Remove DOFs and corresponding indices if they exist
        if dof_1 in self.fixedIndices and dof_2 in self.fixedIndices and dof_3 in self.fixedIndices:
            # Find their indices in fixedDOFs
            index_1 = self.fixedIndices.index(dof_1)
            index_2 = self.fixedIndices.index(dof_2)
            index_3 = self.fixedIndices.index(dof_3)

            # Remove the DOFs from fixedDOFs and corresponding values from fixedIndices
            del self.fixedIndices[index_3]
            del self.fixedIndices[index_2]  # Remove larger index first
            del self.fixedIndices[index_1]
            del self.fixedDOFs[index_3]
            del self.fixedDOFs[index_2]
            del self.fixedDOFs[index_1]

            # Add the DOFs back to freeIndices
            self.freeIndices.append(dof_1)
            self.freeIndices.append(dof_2)
            self.freeIndices.append(dof_3)

            # Keep freeIndices sorted
            self.freeIndices.sort()

    def getBoundaryConditions(self):
        """
        Returns the current boundary conditions as a tuple.

        Returns:
            tuple: (fixedDOFs, fixedIndices, freeIndices)
        """
        return self.fixedDOFs, np.array(self.fixedIndices), self.freeIndices
    

def fun_X0toU(x0_1, x0_2, x0_3, tol=1e-8):
    # Assigna disp BC to DOFs
    if abs(x0_1 - 0.0) < tol:
        u0x1=0.0
        u0x2=0.0
        u0x3=0.0
        return u0x1,u0x2,u0x3
    elif abs(x0_1 - 0.1) < tol:
        u0x1=x0_1*0.1
        u0x2=0.0
        u0x3=0.0
        return u0x1,u0x2,u0x3
    else:
        raise ValueError(f"Not defined BC node for x0_1={x0_1}")
    

def fun_BC_evolution_example_5nodes(bc, X0, Nnodes, t, tol=1e-8):
    """
    Update Dirichlet BCs on `bc` (BoundaryConditions3D) at time `t` by:
      1) Finding all nodes whose x ≈ x_min  → clamp (ux,uy,uz)=(0,0,0)
                 whose x ≈ x_max  → ramp ux=0.1*x0 * t, uy=uz=0
      2) Pinning y- and z-DOFs of all nodes whose y ≈ y_min or y ≈ y_max.
    ----------
    bc : BoundaryConditions3D
        the BC manager to update in-place
    X0 : (3*Nnodes,) array
        original nodal positions [x0,y0,z0, x1,y1,z1, …]
    Nnodes : int
        number of nodes
    t : float
        current time
    tol : float
        tolerance for floating comparisons
    """
    # first pull out all the x‐coords and find their min/max
    x_coords = X0[0::3]
    x_min, x_max = x_coords.min(), x_coords.max()
    left_ids  = np.where(np.isclose(x_coords, x_min, atol=tol))[0]
    right_ids = np.where(np.isclose(x_coords, x_max, atol=tol))[0]

    for node_id in left_ids:
        ux = uy = uz = 0.0
        bc.setBoundaryConditionNode(node_id, np.array([ux, uy, uz]))
    
    for node_id in right_ids:
        x0 = x_coords[node_id]
        du = 0.1*x0
        ux = du*t
        bc.setBoundaryConditionNode(node_id, np.array([ux, uy, uz]))
        

    # 2) then pin y/z of the top/bottom (as before)…
    y_coords = X0[1::3]
    y_min, y_max = y_coords.min(), y_coords.max()
    bottom_ids = np.where(np.isclose(y_coords, y_min, atol=tol))[0]
    top_ids    = np.where(np.isclose(y_coords, y_max, atol=tol))[0]
    
    for node_id in np.concatenate([bottom_ids, top_ids]):
        # for each such node, pin both y (local=1) and z (local=2)
        for local in (1, 2):
            dof = 3*node_id + local
            if dof not in bc.fixedIndices:
                bc.fixedIndices.append(int(dof))
                bc.fixedDOFs.append(0.0)
            if dof in bc.freeIndices:
                bc.freeIndices.remove(int(dof))
                
    bc.freeIndices.sort()


def fun_BC_evolution_2D_truss_TenComp(bc, X0, Nnodes, t, tol=1e-8):
    """
    Update Dirichlet BCs on `bc` (BoundaryConditions3D) at time `t` by:
      1) Clamping nodes at x ≈ x_min → (ux,uy,uz)=(0,0,0)
         Ramping nodes at x ≈ x_max → ux=0.1*x0 * t, uy=0, uz=0
      2) Pinning z‐DOF = 0 for all remaining nodes (x & y free).
    """
    # first pull out all the x‐coords and find their min/max
    x_coords = X0[0::3]
    x_min, x_max = x_coords.min(), x_coords.max()
    left_ids  = np.where(np.isclose(x_coords, x_min, atol=tol))[0]
    right_ids = np.where(np.isclose(x_coords, x_max, atol=tol))[0]

    for node_id in left_ids:
        ux = uy = uz = 0.0
        bc.setBoundaryConditionNode(node_id, np.array([ux, uy, uz]))
    
    for node_id in right_ids:
        x0 = x_coords[node_id]
        du = 0.1*x0
        ux = du*t
        bc.setBoundaryConditionNode(node_id, np.array([ux, uy, uz]))
        

    # pin z‐DOF = 0 on every node not in left_id or right_ids
    all_ids   = set(range(Nnodes))
    fixed_ids = set(left_ids) | set(right_ids)
    other_ids = all_ids - fixed_ids

    for node_id in other_ids:
        dof_z = 3*node_id + 2
        # add to fixedIndices/ fixedDOFs if not already there
        if dof_z not in bc.fixedIndices:
            bc.fixedIndices.append(int(dof_z))
            bc.fixedDOFs.append(0.0)
        # remove from freeIndices
        if dof_z in bc.freeIndices:
            bc.freeIndices.remove(int(dof_z))
                            
    bc.freeIndices.sort()


def fun_BC_evolution_2D_truss_SimpleShear(bc, X0, Nnodes, t, tol=1e-8):
    """
    Update Dirichlet BCs on `bc` at time `t` for a 2D truss:
      - Top edge (y≈y_max): ux = du*t, uy=0, uz=0
      - Bottom edge (y≈y_min): ux=uy=uz=0
      - All others: uz=0
    """
    # extract y‐coordinates
    y_coords = X0[1::3]
    y_min, y_max = y_coords.min(), y_coords.max()
    bottom_ids = np.where(np.isclose(y_coords, y_min, atol=tol))[0]
    top_ids    = np.where(np.isclose(y_coords, y_max, atol=tol))[0]

    # 1) bottom nodes: fully clamp
    for nid in bottom_ids:
        bc.setBoundaryConditionNode(nid, np.array([0.0, 0.0, 0.0]))

    # 2) top nodes: prescribed ux, clamp uy,uz
    for nid in top_ids:
        x0 = X0[3*nid]          # the original x‐coord of this node
        du = 0.1 * y_max           # ramp rate
        ux = du * t             # displacement at time t
        bc.setBoundaryConditionNode(nid, np.array([ux, 0.0, 0.0]))

    # 3) all other nodes: clamp uz only
    all_ids   = set(range(Nnodes))
    fixed_ids = set(bottom_ids) | set(top_ids)
    other_ids = all_ids - fixed_ids

    for nid in other_ids:
        dof_z = 3*nid + 2
        # if not already clamped, clamp the z‐DOF
        if dof_z not in bc.fixedIndices:
            bc.fixedIndices.append(int(dof_z))
            bc.fixedDOFs.append(0.0)
        # and remove from freeIndices if present
        if dof_z in bc.freeIndices:
            bc.freeIndices.remove(int(dof_z))

    # keep freeIndices sorted
    bc.freeIndices.sort()


def fun_BC_3D_hold_corner(bc, X0, Nnodes, x_thresh, y_thresh, tol=1e-8):
    """
    Clamp ux=uy=uz=0 on all nodes with x < x_thresh AND y < y_thresh.
    
    Parameters
    ----------
    bc : BoundaryConditions3D
      your BC manager
    X0 : (3*Nnodes,) array
      flattened original nodal coords [x0,y0,z0, x1,y1,z1, …]
    Nnodes : int
      number of nodes
    x_thresh : float
      threshold in x‐direction
    y_thresh : float
      threshold in y‐direction
    tol : float
      tolerance for floating‐point compare
    """
    # unpack all x‐ and y‐coordinates
    x_coords = X0[0::3]
    y_coords = X0[1::3]

    # find nodes in the lower‐left corner
    mask = (x_coords < x_thresh + tol) & (y_coords < y_thresh + tol)
    node_ids = np.where(mask)[0]

    # clamp all 3 DOFs on each such node
    for nid in node_ids:
        bc.setBoundaryConditionNode(nid, np.array([0.0, 0.0, 0.0]))


def fun_BC_3D_hold_center(
    bc,
    X0: np.ndarray,
    Nnodes: int,
    half_x: float,
    half_y: float,
    tol: float = 1e-8
):
    """
    Clamp ux=uy=uz=0 on all nodes whose (x,y) lies within 
    [x_mid - half_x, x_mid + half_x] × [y_mid - half_y, y_mid + half_y].

    Parameters
    ----------
    bc : BoundaryConditions3D
        your BC manager
    X0 : (3*Nnodes,) array
        flattened original nodal coords [x0,y0,z0, x1,y1,z1, …]
    Nnodes : int
        number of nodes
    half_x : float
        half‐width of the box in x about the mesh center
    half_y : float
        half‐height of the box in y about the mesh center
    tol : float
        tolerance for floating‐point compare
    """
    x = X0[0::3]
    y = X0[1::3]

    # center
    x_mid = 0.5*(x.min() + x.max())
    y_mid = 0.5*(y.min() + y.max())

    # find nodes
    mask = (
        (np.abs(x - x_mid) <= half_x + tol) &
        (np.abs(y - y_mid) <= half_y + tol)
    )
    node_ids = np.nonzero(mask)[0]

    # clamp all 3 DOFs
    for nid in node_ids:
        bc.setBoundaryConditionNode(nid, np.array([0.0, 0.0, 0.0]))
        
    return node_ids


def fun_BC_spoon(
    bc,
    X0: np.ndarray,
    Nnodes: int,
    *,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    node_region_fn: Callable[[np.ndarray], np.ndarray] = None,
    tol: float = 1e-8
) -> np.ndarray:
    """
    Clamp ux=uy=uz=0 on all nodes whose (x,y) lies in the
    rectangle [x_min, x_max] × [y_min, y_max] AND (if provided)
    also satisfies node_region_fn at the node's (x,y).
    
    Returns the array of node‐indices clamped.
    """
    # unpack nodal coords into an (Nnodes,3) array
    nodes = np.empty((Nnodes,3), float)
    nodes[:,0] = X0[0::3]
    nodes[:,1] = X0[1::3]
    nodes[:,2] = X0[2::3]

    # build the rectangle mask
    mask = np.ones(Nnodes, dtype=bool)
    if x_min is not None:
        mask &= (nodes[:,0] >= x_min - tol)
    if x_max is not None:
        mask &= (nodes[:,0] <= x_max + tol)
    if y_min is not None:
        mask &= (nodes[:,1] >= y_min - tol)
    if y_max is not None:
        mask &= (nodes[:,1] <= y_max + tol)

    # if you gave us a second mask function, apply it too
    if node_region_fn is not None:
        # node_region_fn should take an (Nnodes,3) array and return a boolean (Nnodes,) array
        mask &= node_region_fn(nodes)

    # which indices survive?
    node_ids = np.nonzero(mask)[0]

    # clamp all 3 DOFs for each
    for nid in node_ids:
        bc.setBoundaryConditionNode(nid, np.array([0.0, 0.0, 0.0]))

    return node_ids

def fun_BC_spoon_sym(
    bc,
    X0: np.ndarray,
    Nnodes: int,
    *,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    node_region_fn: Callable[[np.ndarray], np.ndarray] = None,
    tol: float = 1e-8
) -> np.ndarray:
    """
    1) Clamp (ux,uy,uz)=0 on nodes in [x_min..x_max]×[y_min..y_max] AND
       (if provided) node_region_fn(nodes)==True.

    2) Then, for any node with x ≈ x_max, also pin its ux=0 (leave uy,uz free).

    Returns the array of *all* node‐indices that got at least one constraint.
    """
    # unpack nodal coords into an (Nnodes,3) array
    nodes = np.empty((Nnodes,3), float)
    nodes[:,0] = X0[0::3]
    nodes[:,1] = X0[1::3]
    nodes[:,2] = X0[2::3]

    # build the rectangle mask
    mask = np.ones(Nnodes, dtype=bool)
    if x_min is not None:
        mask &= (nodes[:,0] >= x_min - tol)
    if x_max is not None:
        mask &= (nodes[:,0] <= x_max + tol)
    if y_min is not None:
        mask &= (nodes[:,1] >= y_min - tol)
    if y_max is not None:
        mask &= (nodes[:,1] <= y_max + tol)

    # optionally intersect with your custom shape‐mask
    if node_region_fn is not None:
        mask &= node_region_fn(nodes)

    # first: clamp all 3 DOFs on those nodes
    clamped_nodes = list(np.nonzero(mask)[0])
    for nid in clamped_nodes:
        bc.setBoundaryConditionNode(nid, np.array([0.0, 0.0, 0.0]))

    # now: if x_max is set, also pin ux on *all* nodes with x ≈ x_max
    if x_max is not None:
        right_nodes = np.nonzero(np.abs(nodes[:,0] - x_max) <= tol)[0]
        for nid in right_nodes:
            dof_x = 3*nid
            # skip if already pinned
            if dof_x in bc.fixedIndices:
                continue
            # otherwise pin just ux
            bc.fixedIndices.append(dof_x)
            bc.fixedDOFs.append(0.0)
            bc.freeIndices.remove(dof_x)

        # include them in your return list too
        clamped_nodes = sorted(set(clamped_nodes) | set(right_nodes))

    return np.array(clamped_nodes, dtype=int)


def fun_BC_peanut(
    bc,
    X0: np.ndarray,
    Nnodes: int,
    *,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    node_region_fn: Callable[[np.ndarray], np.ndarray] = None,
    tol: float = 1e-8
) -> np.ndarray:
    """
    1) Clamp (ux,uy,uz)=0 on nodes in [x_min..x_max]×[y_min..y_max] AND
       (if provided) node_region_fn(nodes)==True.

    2) Then, for any node with x ≈ x_max, also pin its ux=0 (leave uy,uz free).

    Returns the array of *all* node‐indices that got at least one constraint.
    """
    # unpack nodal coords into an (Nnodes,3) array
    nodes = np.empty((Nnodes,3), float)
    nodes[:,0] = X0[0::3]
    nodes[:,1] = X0[1::3]
    nodes[:,2] = X0[2::3]

    # build the rectangle mask
    mask = np.ones(Nnodes, dtype=bool)
    if x_min is not None:
        mask &= (nodes[:,0] >= x_min - tol)
    if x_max is not None:
        mask &= (nodes[:,0] <= x_max + tol)
    if y_min is not None:
        mask &= (nodes[:,1] >= y_min - tol)
    if y_max is not None:
        mask &= (nodes[:,1] <= y_max + tol)

    # optionally intersect with your custom shape‐mask
    if node_region_fn is not None:
        mask &= node_region_fn(nodes)

    # first: clamp all 3 DOFs on those nodes
    clamped_nodes = list(np.nonzero(mask)[0])
    for nid in clamped_nodes:
        bc.setBoundaryConditionNode(nid, np.array([0.0, 0.0, 0.0]))

    return np.array(clamped_nodes, dtype=int)


def fun_BC_beam_PointLoad(bc, X0, Nnodes,
                t=None,
                x_thresh=0.21,
                disp_right=-0.1,
                tol=1e-8):
    """
    Dirichlet BCs:
      1) Clamp all DOFs (ux=uy=uz=0) on nodes with x<x_thresh
      2) On the right‑end (x≈x_max), prescribe uy=disp_right (leave ux,uz free)
    """
    # extract nodal x–coordinates
    x_coords = X0[0::3]
    x_min, x_max = x_coords.min(), x_coords.max()

    # 1) fully clamp left‑end
    left_ids = np.where(x_coords < x_thresh + tol)[0]
    for nid in left_ids:
        bc.setBoundaryConditionNode(nid, np.array([0.0, 0.0, 0.0]))

    # 2) find right‑end nodes
    right_ids = np.where(np.isclose(x_coords, x_max, atol=tol))[0]
    for nid in right_ids:
        # only clamp y‐DOF to disp_right
        dof_y = 3*nid + 1
        if dof_y not in bc.fixedIndices:
            bc.fixedIndices.append(int(dof_y))
            bc.fixedDOFs.append(float(0.0))
        if dof_y in bc.freeIndices:
            bc.freeIndices.remove(int(dof_y))
            
        dof_z = 3*nid + 2 
        if dof_z not in bc.fixedIndices:
            bc.fixedIndices.append(int(dof_z))
            bc.fixedDOFs.append(float(disp_right))
        if dof_z in bc.freeIndices:
            bc.freeIndices.remove(int(dof_z))
            
    # sort freeIndices for cleanliness
    bc.freeIndices.sort()


def fun_BC_beam_FixLeftEnd(bc, X0, x_thresh=0.21, tol=1e-8):
    """
    Dirichlet BCs:
      1) Clamp all DOFs (ux=uy=uz=0) on nodes with x<x_thresh
      2) All other nodes remain free.
    """
    # extract nodal x–coordinates
    x_coords = X0[0::3]

    # 1) fully clamp left‑end
    left_ids = np.where(x_coords < x_thresh + tol)[0]
    for nid in left_ids:
        bc.setBoundaryConditionNode(nid, np.array([0.0, 0.0, 0.0]))

    # sort freeIndices for cleanliness
    bc.freeIndices.sort()
    

def fun_BC_4nodes(bc):

    left_ids  = [0, 1]
    right_ids = [3]

    for node_id in left_ids:
        ux = uy = uz = 0.0
        bc.setBoundaryConditionNode(node_id, np.array([ux, uy, uz]))
    
    for node_id in right_ids:
        ux = uz = 0.0
        uy = 0.00
        bc.setBoundaryConditionNode(node_id, np.array([ux, uy, uz]))
                        
    bc.freeIndices.sort()