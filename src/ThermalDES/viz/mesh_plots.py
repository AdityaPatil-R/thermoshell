import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

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

def plot_truss_3d_path_history(
    q_history: list,
    connectivity,
    highlight_node_id: int,
    NP_total=None,
    title=None,
    figsize=(8, 8)
):
    """
    Plots the full deformation path of a truss.

    - Draws the initial state (gray).
    - Draws the final state (black).
    - Draws the full path of a single highlighted node (red line).

    Parameters
    ----------
    q_history : list of np.ndarray
        A list where each element is a (Ndofs,) coordinate vector 'q'
        from each step of the simulation.
    connectivity : array-like, shape (Nedges,3)
        Each row [eid, node_i, node_j].
    highlight_node_id : int
        The 0-based index of the node to track.
    NP_total : int, optional
        Number of nodes. If None, inferred from q_history[0].
    title : str, optional
        Title for the plot.
    figsize : tuple, optional
        Figure size.
    """

    if not q_history:
        print("Warning: q_history is empty. Nothing to plot.")
        return

    if NP_total is None:
        NP_total = q_history[0].size // 3

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # --- Helper function to draw one mesh state ---
    def _plot_mesh_state(q, color, lw, alpha):
        for eid, i0, i1 in connectivity.astype(int):
            p0 = q[3 * i0: 3 * i0 + 3]
            p1 = q[3 * i1: 3 * i1 + 3]
            ax.plot([p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color=color, lw=lw, alpha=alpha)
    # --- End helper ---

    # 1. Plot the initial mesh (step 0)
    _plot_mesh_state(q_history[0], color='gray', lw=1, alpha=0.5)

    # 2. Plot the final mesh (last step)
    _plot_mesh_state(q_history[-1], color='black', lw=1, alpha=1.0)

    # 3. Extract and plot the path of the highlighted node
    path = []
    for q in q_history:
        # Get the [x, y, z] coords for the highlighted node
        p = q[3 * highlight_node_id: 3 * highlight_node_id + 3]
        path.append(p)
    path = np.array(path)

    # Plot the continuous path
    ax.plot(path[:, 0], path[:, 1], path[:, 2],
            color='red', lw=2.5)

    # 4. Highlight the start and end points of that path
    ax.scatter(*path[0], color='blue', s=80,
               label='Start', marker='o', depthshade=False, zorder=10)
    ax.scatter(*path[-1], color='red', s=80,
               label='End', marker='X', depthshade=False, zorder=10)

    if title:
        ax.set_title(title)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    
    # Autoscale based on the *entire* path plus the start/end meshes
    all_q = np.concatenate([q_history[0], q_history[-1]])
    all_x = np.concatenate([all_q[0::3], path[:, 0]])
    all_y = np.concatenate([all_q[1::3], path[:, 1]])
    all_z = np.concatenate([all_q[2::3], path[:, 2]])
    
    # Simple bounding box
    max_range = np.array([all_x.max()-all_x.min(), 
                          all_y.max()-all_y.min(), 
                          all_z.max()-all_z.min()]).max() / 2.0

    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
    return ax

def plot_truss_3d_history_overlay(
    q_history: List[np.ndarray],
    connectivity,
    NP_total=None,
    title=None,
    figsize=(8, 8),
    start_color='gray',
    end_color='black',
    trans_color='blue',
    trans_alpha=0.05,
    lw=1.0,
    max_trans_frames=20
):
    """
    Plots an overlay of the truss deformation history.

    - Draws the initial state (start_color).
    - Draws the final state (end_color).
    - Draws a selection of intermediate states with high transparency
      (trans_color, trans_alpha) to create a "volume" or "motion blur" effect.

    Parameters
    ----------
    q_history : list of np.ndarray
        A list where each element is a (Ndofs,) coordinate vector 'q'
        from each step of the simulation.
    connectivity : array-like, shape (Nedges,3)
        Each row [eid, node_i, node_j].
    NP_total : int, optional
        Number of nodes. If None, inferred from q_history[0].
    title : str, optional
        Title for the plot.
    figsize : tuple, optional
        Figure size.
    start_color : str, optional
        Color for the first (t=0) mesh.
    end_color : str, optional
        Color for the last (t=final) mesh.
    trans_color : str, optional
        Color for the intermediate "transition" meshes.
    trans_alpha : float, optional
        Alpha (transparency) for the intermediate meshes.
    lw : float, optional
        Line width for all mesh plots.
    max_trans_frames : int, optional
        The maximum number of intermediate frames to draw. This prevents
        the plot from becoming too slow or cluttered if q_history is large.
    """

    if not q_history or len(q_history) < 2:
        print("Warning: q_history must contain at least 2 states to plot an overlay.")
        return

    if NP_total is None:
        NP_total = q_history[0].size // 3

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    def _plot_mesh_state(q, color, lw, alpha):
        for eid, i0, i1 in connectivity.astype(int):
            p0 = q[3 * i0: 3 * i0 + 3]
            p1 = q[3 * i1: 3 * i1 + 3]
            ax.plot([p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color=color, lw=lw, alpha=alpha, zorder=1)
    
    intermediate_states = q_history[1:-1]
    n_intermediate = len(intermediate_states)

    if n_intermediate > 0:
        if n_intermediate > max_trans_frames:
            indices = np.linspace(0, n_intermediate - 1, max_trans_frames).astype(int)
            states_to_plot = [intermediate_states[i] for i in indices]
        else:
            states_to_plot = intermediate_states
            
        print(f"  Plotting {len(states_to_plot)} intermediate states...")
        for q in states_to_plot:
            _plot_mesh_state(q, color=trans_color, lw=lw, alpha=trans_alpha)

    print("  Plotting initial state...")
    _plot_mesh_state(q_history[0], color=start_color, lw=lw, alpha=0.7)

    print("  Plotting final state...")
    _plot_mesh_state(q_history[-1], color=end_color, lw=lw, alpha=1.0)


    if title:
        ax.set_title(title)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    all_q = np.concatenate(q_history)
    all_x = all_q[0::3]
    all_y = all_q[1::3]
    all_z = all_q[2::3]
    
    # Simple bounding box
    max_range = np.array([all_x.max()-all_x.min(), 
                          all_y.max()-all_y.min(), 
                          all_z.max()-all_z.min()]).max() / 2.0

    if max_range == 0: # Handle case where nothing moved
        max_range = 1.0

    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
    return ax