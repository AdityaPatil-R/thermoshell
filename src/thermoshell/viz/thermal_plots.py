import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

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