import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_truss_3d(q, 
                  connectivity, 
                  NP_total=None, 
                  title=None,
                  figsize=(6,6), 
                  show_labels=True):
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

    q = np.asarray(q)

    if NP_total is None:
        NP_total = q.size // 3

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, 
                          projection='3d')

    # Draw each edge in gray and label its ID
    for eid, i0, i1 in connectivity.astype(int):
        p0 = q[3*i0 : 3*i0+3]
        p1 = q[3*i1 : 3*i1+3]
        ax.plot([p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color='gray', 
                lw=1)
        
        if show_labels:
            mid = (p0 + p1) * 0.5
            ax.text(*mid, 
                    str(eid),
                    color='blue', 
                    fontsize=9,
                    ha='center', 
                    va='center')

    # Scatter & label each node in pink
    for nid in range(NP_total):
        p = q[3*nid : 3*nid+3]
        ax.scatter(*p, 
                   color='pink', 
                   s=30)

        if show_labels:
            ax.text(*p, 
                    str(nid), 
                    color='black', 
                    fontsize=9,
                    ha='center', 
                    va='center')

    if title:
        ax.set_title(title)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Autoscale
    ax.auto_scale_xyz(q[0::3], 
                      q[1::3], 
                      q[2::3])
    plt.show()

    return ax


def plot_truss_2d(q, 
                  connectivity, 
                  NP_total=None, 
                  title=None, 
                  figsize=(6,6),
                  show_labels=True):
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

    # Decide stride and number of nodes
    if L % 3 == 0:
        stride = 3
    elif L % 2 == 0:
        stride = 2
    else:
        raise ValueError("q length must be multiple of 2 or 3.")
    
    N = L // stride if NP_total is None else NP_total

    # Extract x,y
    x = q[0 :: stride]
    y = q[1 :: stride]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    # Draw edges and label
    for eid, n1, n2 in connectivity.astype(int):
        x0, y0 = x[n1], y[n1]
        x1, y1 = x[n2], y[n2]
        ax.plot([x0, x1], 
                [y0, y1], 
                color='gray', 
                lw=1)
        
        if show_labels:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, 
                    my, 
                    str(eid),
                    fontsize=10, 
                    color='blue',
                    ha='center', 
                    va='center')

    # Draw nodes and label
    for nid in range(N):
        ax.scatter(x[nid], 
                   y[nid], 
                   color='pink', 
                   s=30)
        
        if show_labels:
            ax.text(x[nid], 
                    y[nid], 
                    str(nid),
                    fontsize=10, 
                    color='black',
                    ha='center', 
                    va='center')

    if title:
        ax.set_title(title)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

    return ax