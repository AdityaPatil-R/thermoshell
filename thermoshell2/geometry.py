import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple, Dict
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

def load_mesh(filename):
    nodeXYZ      = []
    Triangles    = []
    Connectivity = []

    section = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # detect new section
            if line.startswith('*'):
                if 'shellNodes' in line:
                    section = 'nodes'
                elif 'FaceNodes' in line:
                    section = 'triangles'
                elif 'Edges' in line:
                    section = 'edges'
                else:
                    section = None
                continue

            # parse a data line
            parts = [p.strip() for p in line.split(',')]
            if section == 'nodes':
                # node ID, x, y, z
                nid = int(parts[0])
                x, y, z = map(float, parts[1:4])
                nodeXYZ.append([nid, x, y, z])

            elif section == 'triangles':
                # element ID, node1, node2, node3
                eid = int(parts[0])
                n1, n2, n3 = map(int, parts[1:4])
                Triangles.append([eid, n1, n2, n3])

            elif section == 'edges':
                # edge ID, left node, right node
                eid = int(parts[0])
                n1, n2 = map(int, parts[1:3])
                Connectivity.append([eid, n1, n2])

    # convert to NumPy arrays
    nodeXYZ      = np.array(nodeXYZ,      dtype=float)
    Triangles    = np.array(Triangles,    dtype=int)
    Connectivity = np.array(Connectivity, dtype=int)

    return nodeXYZ, Connectivity, Triangles


def fun_edge_lengths(nodeXYZ: np.ndarray,
                         Connectivity: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean length of each edge.

    Parameters
    ----------
    nodeXYZ : array, shape (Nnodes, 4)
        Columns are [nodeID, x, y, z].
    Connectivity : array, shape (Nedges, 3)
        Columns are [edgeID, node_i, node_j].

    Returns
    -------
    edge_lengths : array, shape (Nedges,)
        edge_lengths[k] is the length of the edge whose ID is k.
    """
    coords = nodeXYZ[:, 1:4]     # shape (Nnodes, 3)

    Nedges = Connectivity.shape[0]
    edge_lengths = np.zeros(Nedges)

    for eid, ni, nj in Connectivity.astype(int):
        p_i = coords[ni,:]
        p_j = coords[nj,:]
        edge_lengths[eid] = np.linalg.norm(p_j - p_i)

    return edge_lengths


def new_fig(num=None, figsize=(6,6), label_fmt='Fig. {n}',
            label_pos=(0.01, 1.00), **subplot_kw):
    """
    Create (or switch to) figure `num` and a single Axes,
    then write 'Fig. <num>' in the upper‐left corner of the figure.
    Returns (fig, ax).
    
    - num: figure number (int) or None to auto‐increment.
    - figsize: passed through to plt.figure.
    - label_fmt: format string; '{n}' will be replaced by fig.number.
    - label_pos: (x, y) in figure fraction (0–1) for the label.
    - subplot_kw: passed to fig.add_subplot (e.g. projection='3d').
    """
    fig = plt.figure(num, figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1, **subplot_kw)
    
    lbl = label_fmt.format(n=fig.number)
    # place in figure coords, so works for 2D or 3D
    fig.text(
        label_pos[0], label_pos[1], lbl,
        transform=fig.transFigure,
        fontsize=12, fontweight='bold',
        va='top'
    )
    return fig, ax


def plot_truss_3d(q, connectivity, NP_total=None, title=None,
                  figsize=(6,6), show_labels=True):
    """
    Plot a 3D truss given nodal coords `q` and an edge list `connectivity`,
    with an option to turn on/off node and edge labels.

    Parameters
    ----------
    q : array-like, shape (3*Nnodes,)
        Current nodal coordinates [x0,y0,z0, x1,y1,z1, ...].
    connectivity : array-like, shape (Nedges,3)
        Each row [eid, node_i, node_j].
    NP_total : int, optional
        Number of nodes.  If None, inferred as len(q)//3.
    title : str, optional
        Title for the plot.
    figsize : tuple, optional
        Figure size.
    show_labels : bool, optional
        Whether to draw node and edge ID labels.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    q = np.asarray(q)
    if NP_total is None:
        NP_total = q.size // 3

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    # draw each edge in gray and label its ID
    for eid, i0, i1 in connectivity.astype(int):
        p0 = q[3*i0  : 3*i0+3]
        p1 = q[3*i1  : 3*i1+3]
        ax.plot([p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color='gray', lw=1)
        if show_labels:
            mid = (p0 + p1) * 0.5
            ax.text(*mid, str(eid), color='blue', fontsize=9,
                    ha='center', va='center')

    # scatter & label each node in pink
    for nid in range(NP_total):
        p = q[3*nid : 3*nid+3]
        ax.scatter(*p, color='pink', s=30)
        if show_labels:
            ax.text(*p, str(nid), color='black', fontsize=9,
                    ha='center', va='center')

    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # autoscale
    ax.auto_scale_xyz(q[0::3], q[1::3], q[2::3])

    plt.show()
    return ax


def plot_truss_2d(q, connectivity, NP_total=None, title=None, figsize=(6,6),show_labels=True):
    """
    Plot a 2D truss given nodal DOFs `q` and an edge list `connectivity`.

    Parameters
    ----------
    q : array-like, shape (2*Nnodes,) or (3*Nnodes,)
        Current nodal DOFs.  If length is 3*N, we assume [x,y,z] and ignore z.
        If length is 2*N, we take it as [x,y].
    connectivity : array-like, shape (Nedges,3)
        Each row is [eid, node_i, node_j].
    NP_total : int, optional
        Number of nodes.  If None, inferred as len(q)//2 or len(q)//3.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    q = np.asarray(q)
    L = q.size

    # decide stride and number of nodes
    if L % 3 == 0:
        stride = 3
    elif L % 2 == 0:
        stride = 2
    else:
        raise ValueError("q length must be multiple of 2 or 3.")
    N = L // stride if NP_total is None else NP_total

    # extract x,y
    x = q[0::stride]
    y = q[1::stride]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    # draw edges + label
    for eid, n1, n2 in connectivity.astype(int):
        x0, y0 = x[n1], y[n1]
        x1, y1 = x[n2], y[n2]
        ax.plot([x0, x1], [y0, y1], color='gray', lw=1)
        if show_labels:
            mx, my = (x0 + x1)/2, (y0 + y1)/2
            ax.text(mx, my, str(eid),
                    fontsize=10, color='blue',
                    ha='center', va='center')

    # draw nodes + label
    for nid in range(N):
        ax.scatter(x[nid], y[nid], color='pink', s=30)
        if show_labels:
            ax.text(x[nid], y[nid], str(nid),
                    fontsize=10, color='black',
                    ha='center', va='center')

    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return ax


def plot_thermal_strain_edges(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    epsilon_th: np.ndarray,
    title: str = None,
    figsize: tuple = (6,6),
    cmap: str = 'viridis'
):
    """
    Plot a 2D truss mesh with each edge colored by its thermal strain.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
        The nodal coordinates [x,y,z] (we only use x,y).
    connectivity : (Nedges,3) array
        Each row [eid, node_i, node_j].  We assume eid runs 0..Nedges-1.
    epsilon_th : (Nedges,) array
        Thermal strain assigned to each edge.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    cmap : str
        A Matplotlib colormap name.
    """
    # Extract x,y
    x = node_coords[:,0]
    y = node_coords[:,1]

    # Build segment list and corresponding strain values
    segments = []
    values   = []
    for eid, n0, n1 in connectivity.astype(int):
        segments.append([(x[n0], y[n0]), (x[n1], y[n1])])
        values.append(epsilon_th[eid])
    values = np.array(values)

    # Create a LineCollection: one line per edge
    lc = LineCollection(segments,
                        array=values,
                        cmap=plt.get_cmap(cmap),
                        linewidths=2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(lc)
    ax.autoscale()            # adjust view to the data
    ax.set_aspect('equal')    # equal x/y scales

    # add colorbar
    cbar = fig.colorbar(lc, ax=ax, label='Thermal strain εᵗʰ')

    # annotate if you like
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
    return ax


def plot_thermal_strain_edges_CustomRange(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    epsilon_th: np.ndarray,
    title: str = None,
    figsize: tuple = (6,6),
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None
):
    # Extract x,y
    x = node_coords[:,0]
    y = node_coords[:,1]

    # Build segment list and corresponding strain values
    segments = []
    values   = []
    for eid, n0, n1 in connectivity.astype(int):
        segments.append([(x[n0], y[n0]), (x[n1], y[n1])])
        values.append(epsilon_th[eid])
    values = np.array(values)

    # Create a Normalize instance if limits are provided
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Create a LineCollection: one line per edge
    lc = LineCollection(
        segments,
        array=values,
        cmap=plt.get_cmap(cmap),
        norm=norm,
        linewidths=2
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(lc)
    ax.autoscale()            # adjust view to the data
    ax.set_aspect('equal')    # equal x/y scales

    # add colorbar
    cbar = fig.colorbar(lc, ax=ax, label='value')

    # annotate if you like
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
    return ax


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