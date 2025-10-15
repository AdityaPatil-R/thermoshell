
# %% 3D truss with stretch energy. 04-28-25
# %% Dihedral angle studied. 04-30-25
# %% Bending added and verified with EB beam point load. 05-02-25
# Orgnize the code. 05-04-25 11am
# Coupled energy added and hinge quad works. 05-05-25
# Created fun_coupled_Ebend_grad_hess, fun_DEps_grad_hess in _assemble_bending_coupled_v2 of class ElasticGHEdgesCoupled. 05-07-25
# Verified Grad and Hess of coupled energy of whole system. 05-07-25
# Added and verified thermal axial strain.  05-10-25
# Explore patterns of thermal strains...
# Added distribution of Y and ks_array. 05-11-25
# Added kb_array. 05-12-25
# Added asymmetric kb_array values for dihedral angle. 05-12-25
# Added adaptive timestep. 05-17-25
# Non-uniform Ysoft, Yhard, Ysoft_bend, Yhard_bend. 05-19-25
# Added output node_ids from fun_BC_3D_hold_center. 05-20-25
# Added stripe pattern. 05-20-25
# Added rectangular mesh option iMesh=8. 05-20-25
# Added non-uniform thermal axial strains. 05-21-25
# Added gravity force. 05-21-25
# g_vec=-g_vec  # The z axis is inverse of physical world. 05-21-25
# put the ellipse shape thermal strains from plate_unit_FEM_0523_spoon.
# Added non-uniform Y for hard material. 05-24-25
# Added envelop transition of thermal strains. 05-25-25
# Added whole panut shape. 05-26-25
# To add Ks and Kb varying with time...
# Added vanishing of gravity at the end. 05-29-25
# Change error to relative error. 06-03-25
# Added effective kb and ks for bilayer composite. 07-08-25
# Calibrated parameters for experiments. 07-10-25
# Added FactorKs, FactorKb. 07-15-25
# Added cross shape scaled to 100mm. 07-17-25
# Added canoe shape scaled to 150mm. 07-21-25
# To add parallel calculation.



import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import scipy.io as sio
from itertools import combinations
from collections import defaultdict
from typing import List, Callable, Tuple, Dict
from matplotlib.collections import LineCollection
from typing import Callable
from matplotlib.colors import Normalize
from functools import partial
import argparse


start = time.perf_counter()




# --------------------
iPrint = 0
iPlot = 0
iTest = 0
eps_thermal = -0.3
iMesh = 1
iGravity = 1
iFluc = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', type=int, help="which mesh to load")
    p.add_argument('--eps-thermal', type=float, help="thermal strain")
    p.add_argument('--print', dest='do_print', action='store_true')
    p.add_argument('--plot', dest='do_plot', action='store_true')
    p.add_argument('--test', dest='do_test', action='store_true')
    p.add_argument('--gravity', type=int, help="0=no gravity, 1=gravity")
    p.add_argument('--fluctuate', type=int, help="0=no, 1=yes")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mesh is not None:       iMesh = args.mesh
    if args.eps_thermal is not None: eps_thermal = args.eps_thermal
    if args.do_print:               iPrint = 0
    if args.do_plot:                iPlot = 0
    if args.do_test:                iTest = 0
    if args.gravity is not None:    iGravity = args.gravity
    if args.fluctuate is not None:  iFluc = args.fluctuate
    # 

    print("CLI overrides:", {k: v for k, v in vars(args).items() if v is not None and v is not False})


'''
iPrint = 0
iPlot  = 0
iTest  = 0

eps_thermal=-0.05


# Mesh options
iMesh = 3


if iMesh not in mesh_files:
    raise ValueError(f"Invalid iMesh value {iMesh}. Valid options: {list(mesh_files.keys())}")

    
# 0: no gravity force. 1:external nodal force.
iGravity = 1

# 0: no boundary fluctuation in thermal strains; 1: yes for fluctuation.
iFluc = 0
'''





mesh_files = {
    1: 'mesh_python_circle_970nodes_scale100mm.txt',
    2: 'mesh_rectangle_scaled_1215nodes_scale100mm_155mm.txt',
    3: 'mesh_python_square_388nodes_scale100mm.txt',
}


if iMesh == 1:
    Yratio   = 1.0    # Ratio between in and out Y values.
    OuterR   = 0.05   # Outer radius of circular plate.
    # Pattern parameters
    Pattern_center=[0.0,0.0]
    StripeWidth=0.0059   #0.01
    StripeLength=0.0441  #0.07
    Stripe_r=0.0029
    

# scale=0.39817
# Rectangular mesh
if iMesh == 2:
    Yratio   = 1.0
    OuterR   = 0.05
    delta_shape=0.02787
    n_spokes = 6   # Number of arms
    star_radius  = 0.044595,
    star_thickness = 0.002787,
    beam_thickness = 0.002787
    

if iMesh == 3:
    Yratio   = 1.0
    OuterR   = 0.05
    
    Pattern_center=[0.05, 0.05]
    StripeWidth=0.0059
    StripeLength=0.0707
    Stripe_r=0.0006
    
    
# Magnitude percentage of fluctuation
epsilon_th_fluctuation=2.0

eps_thermal_min = 0
eps_thermal_max = eps_thermal

# %% Effective ks and kb for a bilayer composite

def bilayer_flexural_rigidity(h1, h2, Y1, Y2, b1=1.0, b2=1.0):
    """
    Computes the effective flexural rigidity D_eff and the discrete hinge stiffness k_b
    for a bilayer composite.
    """
    # layer centroids (measured from bottom of layer 1)
    y1 = h1 / 2.0
    y2 = h1 + h2 / 2.0

    # neutral axis location
    num = Y1 * b1 * h1 * y1 + Y2 * b2 * h2 * y2
    den = Y1 * b1 * h1     + Y2 * b2 * h2
    ybar = num / den

    # individual layer contributions to D
    D1 = Y1 * b1 * (h1**3 / 12.0 + h1 * (y1 - ybar)**2)
    D2 = Y2 * b2 * (h2**3 / 12.0 + h2 * (y2 - ybar)**2)

    # total effective rigidity and hinge stiffness
    D_eff = D1 + D2

    return D_eff


Ysoft       = 1.0e6
Yhard       = 3.0e6

FactorKs=10.0
FactorKb=1.0   # Let Kb get smaller than geometry defined.


# ------ physical ks and kb. 07-08-2025 ------

 # lk: mean length, h1: thickness of shrinky dink layer, h2: thickness of PLA layer
PARAMS = {
    1: (0.0032, 0.3e-3, 0.7e-3),  # lk, h1,   h2
    2: (0.0040, 0.3e-3, 0.6e-3),
    3: (0.0058, 0.3e-3, 1.0e-3),
}

try:
    lk, h1, h2 = PARAMS[iMesh]
except KeyError:
    raise ValueError(f"Unsupported mesh index: {iMesh!r}") from None



b1=1.0
D_1 = Ysoft*b1*(h1**3)/12.0;
D_12 = bilayer_flexural_rigidity(h1, h2, Ysoft, Yhard)
print(f"Shrinky dink: D_1 = {D_1:.8f} per unit width")
print(f"Bilayer composite: D_12 = {D_12:.8f} per unit width")

kb1  = (2.0 / np.sqrt(3.0)) * D_1
kb12 = (2.0 / np.sqrt(3.0)) * D_12
print(f"kb1   = {kb1:.8f} N·m")
print(f"kb12   = {kb12:.8f} N·m")
print(f"kb12/kb1   = {kb12/kb1:.8f} N·m")


ks1  = Ysoft * h1 * (lk**2) * np.sqrt(3) / 2.0
ks2  = Yhard * h2 * (lk**2) * np.sqrt(3) / 2.0
ks12 = ks1 + ks2
print(f"ks1  = {ks1:.8f} N·m")
print(f"ks2  = {ks2:.8f} N·m")
print(f"ks12 = {ks12:.8f} N·m")



# %% Def. functions for mesh, length, BC, and plot.

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


# def plot_truss_3d(q, connectivity, NP_total=None, title=None, figsize=(6,6)):
#     """
#     Plot a 3D truss given nodal coords `q` and an edge list `connectivity`.

#     Parameters
#     ----------
#     q : array-like, shape (3*Nnodes,)
#         Current nodal coordinates [x0,y0,z0, x1,y1,z1, ...].
#     connectivity : array-like, shape (Nedges,3)
#         Each row [eid, node_i, node_j].
#     NP_total : int, optional
#         Number of nodes.  If None, inferred as len(q)//3.
#     title : str, optional
#         Title for the plot.
#     figsize : tuple, optional
#         Figure size.
#     """
#     if NP_total is None:
#         NP_total = q.size // 3

#     fig = plt.figure(figsize=figsize)
#     ax  = fig.add_subplot(111, projection='3d')

#     # draw each edge in gray and label its ID
#     for eid, i0, i1 in connectivity.astype(int):
#         p0 = q[3*i0  : 3*i0+3]
#         p1 = q[3*i1  : 3*i1+3]
#         ax.plot([p0[0], p1[0]],
#                 [p0[1], p1[1]],
#                 [p0[2], p1[2]],
#                 color='gray', lw=1)
#         mid = (p0 + p1)*0.5
#         ax.text(*mid, str(eid), color='blue')

#     # scatter & label each node in pink
#     for nid in range(NP_total):
#         p = q[3*nid : 3*nid+3]
#         ax.scatter(*p, color='pink', s=30)
#         ax.text(*p, str(nid), color='black')

#     if title:
#         ax.set_title(title)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.auto_scale_xyz(q[0::3], q[1::3], q[2::3])
#     plt.show()


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


# %% Def. functions: stretch strain, gradient d(stretch strain)/dx, hessian d^2(stretch strain)/dx^2.

def get_strain_stretch_edge2D3D(node0, node1, l_k):
    # Works for both 2D and 3D.
    # l_k (float): Reference (undeformed) length of the edge.
    edge = node1 - node0
    edgeLen = np.linalg.norm(edge)
    epsX = edgeLen / l_k - 1
    return epsX


def test_get_strain_stretch_edge2D3D():
  '''
  Verify get_strain_stretch_edge2D3D() function.
  '''
  node0 = np.array([0.0, 0.0, 0.0])
  node1 = np.array([1.0, 1.0, 1.0])
  l_k = 1.5
  stretch = get_strain_stretch_edge2D3D(node0, node1, l_k)
  print("Axial stretch:", stretch)


# test_get_strain_stretch_edge2D3D() # verified


def grad_and_hess_strain_stretch_edge2D(node0, node1, l_k):
  '''
  Compute the gradient and hessian of the axial stretch of a 2D edge with
  respect to the dof vector (4 dofs: x,y coordinates of the two nodes)

  Inputs:
  node0: 2x1 vector - position of the first node
  node1: 2x1 vector - position of the last node
  l_k: reference length (undeformed) of the edge

  Outputs:
  dF: 4x1  vector - gradient of axial stretch between node0 and node 1.
  dJ: 4x4 vector - hessian of axial stretch between node0 and node 1.
  '''

  edge = node1 - node0
  edgeLen = np.linalg.norm(edge)
  tangent = edge / edgeLen  # unit directional vector
  epsX = get_strain_stretch_edge2D3D(node0, node1, l_k)

  dF_unit = tangent / l_k # gradient of stretch with respect to the edge vector
  dF = np.zeros((4))
  dF[0:2] = - dF_unit
  dF[2:4] = dF_unit

  # M (see below) is the Hessian of square(stretch) with respect to the edge vector
  Id3 = np.eye(2)
  M = 2.0 / l_k * ((1 / l_k - 1 / edgeLen) * Id3 + 1 / edgeLen * ( np.outer( edge, edge ) ) / edgeLen ** 2)

  # M is the Hessian of stretch with respect to the edge vector
  if epsX == 0: # Edge case
    M2 = np.zeros_like(M)
  else:
    M2 = 1.0/(2.0*epsX) * (M - 2.0*np.outer(dF_unit,dF_unit))

  dJ = np.zeros((4,4))
  dJ[0:2,0:2] = M2
  dJ[2:4,2:4] = M2
  dJ[0:2,2:4] = - M2
  dJ[2:4,0:2] = - M2

  return dF,dJ


def test_stretch_edge2D():
  '''
  This function tests the outputs (gradient and hessian) of
  grad_and_hess_strain_stretch_edge2D() against their finite difference
  approximations.
  '''

  # Undeformed configuration (2 nodes, 1 edge)
  x0 = np.random.rand(4)
  l_k = np.linalg.norm(x0[0:2] - x0[2:4])

  # Deformed configuration (2 nodes, 1 edge)
  x = np.random.rand(4)
  node0 = x[0:2]
  node1 = x[2:4]

  # Stretching force and Jacobian
  grad, hess = grad_and_hess_strain_stretch_edge2D(node0, node1, l_k)

  # Use FDM to compute Fs and Js
  change = 1e-6
  strain_ground = get_strain_stretch_edge2D3D(node0, node1, l_k)
  grad_fdm = np.zeros(4)
  hess_fdm = np.zeros((4,4))
  for c in range(4):
    x_plus = x.copy()
    x_plus[c] += change
    node0_plus = x_plus[0:2]
    node1_plus = x_plus[2:4]
    strain_plus = get_strain_stretch_edge2D3D(node0_plus, node1_plus, l_k)
    grad_fdm[c] = (strain_plus - strain_ground) / change

    grad_change, _ = grad_and_hess_strain_stretch_edge2D(node0_plus, node1_plus, l_k)
    hess_fdm[:,c] = (grad_change - grad) / change

  # First plot: Forces (Gradients)
  plt.figure(1)  # Equivalent to MATLAB's figure(1)
  plt.plot(grad, 'ro', label='Analytical')  # Plot analytical forces
  plt.plot(grad_fdm, 'b^', label='Finite Difference')  # Plot finite difference forces
  plt.legend(loc='best')  # Display the legend
  plt.xlabel('Index')  # X-axis label
  plt.ylabel('Gradient')  # Y-axis label
  plt.title('Forces (Gradients) Comparison')  # Optional title
  plt.show()  # Show the plot

  # Second plot: Hessians
  plt.figure(2)  # Equivalent to MATLAB's figure(2)
  plt.plot(hess.flatten(), 'ro', label='Analytical')  # Flatten and plot analytical Hessian
  plt.plot(hess_fdm.flatten(), 'b^', label='Finite Difference')  # Flatten and plot finite difference Hessian
  plt.legend(loc='best')  # Display the legend
  plt.xlabel('Index')  # X-axis label
  plt.ylabel('Hessian')  # Y-axis label
  plt.title('Hessian Comparison')  # Optional title
  plt.show()  # Show the plot
  return

# test_stretch_edge2D()


# Update formulas with zero stretch for nonzero stiffness component. 

def grad_and_hess_strain_stretch_edge3D_ZeroStrainStiff(node0, node1, l_k, tol=1e-10):
    '''
    Compute the gradient and Hessian of the axial stretch of a 3D edge with
    respect to the DOF vector (6 DOFs: x,y,z coords of the two nodes).

    Inputs:
      node0: length-3 array – position of the first node [x0,y0,z0]
      node1: length-3 array – position of the second node [x1,y1,z1]
      l_k:    float        – reference (undeformed) length of the edge

    Outputs:
      dF: length-6 array   – gradient of stretch w.r.t. [x0,y0,z0,x1,y1,z1]
      dJ: 6×6 array        – Hessian of stretch
    '''
    # edge vector and its length
    edge    = node1 - node0
    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen

    # axial stretch
    epsX = get_strain_stretch_edge2D3D(node0, node1, l_k)

    # gradient of stretch w.r.t. edge-vector
    dF_unit = tangent / l_k
    dF = np.zeros(6)
    dF[0:3] = -dF_unit
    dF[3:6] =  dF_unit

    # Hessian of squared-stretch w.r.t. edge-vector (3×3)
    I3 = np.eye(3)
    M  = 2.0 / l_k * (
           (1.0/l_k - 1.0/edgeLen) * I3
         + (1.0/edgeLen) * np.outer(edge, edge) / edgeLen**2
        )

    # convert to Hessian of stretch itself
    if abs(epsX) < tol:
        # small‐strain limit: (I - t t^T)/(L0 * L)
        M2 = (I3 - np.outer(tangent, tangent)) / (l_k * edgeLen)
    else:
        # full nonlinear Hessian of ε
        M2 = 1.0/(2.0*epsX) * (M - 2.0*np.outer(dF_unit, dF_unit))
        
    # assemble 6×6 Hessian
    dJ = np.zeros((6,6))
    dJ[ 0:3,  0:3] =  M2
    dJ[ 3:6,  3:6] =  M2
    dJ[ 0:3,  3:6] = -M2
    dJ[ 3:6,  0:3] = -M2

    return dF, dJ


def grad_and_hess_strain_stretch_edge3D(node0, node1, l_k):
    '''
    Compute the gradient and Hessian of the axial stretch of a 3D edge with
    respect to the DOF vector (6 DOFs: x,y,z coords of the two nodes).

    Inputs:
      node0: length-3 array – position of the first node [x0,y0,z0]
      node1: length-3 array – position of the second node [x1,y1,z1]
      l_k:    float        – reference (undeformed) length of the edge

    Outputs:
      dF: length-6 array   – gradient of stretch w.r.t. [x0,y0,z0,x1,y1,z1]
      dJ: 6×6 array        – Hessian of stretch
    '''
    # edge vector and its length
    edge    = node1 - node0
    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen

    # axial stretch
    epsX = get_strain_stretch_edge2D3D(node0, node1, l_k)

    # gradient of stretch w.r.t. edge-vector
    dF_unit = tangent / l_k
    dF = np.zeros(6)
    dF[0:3] = -dF_unit
    dF[3:6] =  dF_unit

    # Hessian of squared-stretch w.r.t. edge-vector (3×3)
    I3 = np.eye(3)
    M  = 2.0 / l_k * (
           (1.0/l_k - 1.0/edgeLen) * I3
         + (1.0/edgeLen) * np.outer(edge, edge) / edgeLen**2
        )

    # convert to Hessian of stretch itself
    if epsX == 0.0:
        M2 = np.zeros_like(M)
        
    else:
        M2 = 1.0/(2.0*epsX) * (M - 2.0*np.outer(dF_unit, dF_unit))

    # assemble 6×6 Hessian
    dJ = np.zeros((6,6))
    dJ[ 0:3,  0:3] =  M2
    dJ[ 3:6,  3:6] =  M2
    dJ[ 0:3,  3:6] = -M2
    dJ[ 3:6,  0:3] = -M2

    return dF, dJ


def test_stretch_edge3D(iPlot):
    '''
    Finite‐difference check of gradient & Hessian in 3D.
    '''
    # random 6-vector: two nodes
    x0 = np.random.rand(6)
    # reference length from the “undeformed” config
    l_k = np.linalg.norm(x0[0:3] - x0[3:6])

    # “current” config
    x = np.random.rand(6)
    node0 = x[0:3]
    node1 = x[3:6]

    # analytical
    grad, hess = grad_and_hess_strain_stretch_edge3D(node0, node1, l_k)

    # finite‐difference
    change    = 1e-6
    strain0   = get_strain_stretch_edge2D3D(node0, node1, l_k)
    grad_fdm  = np.zeros(6)
    hess_fdm  = np.zeros((6,6))

    for c in range(6):
        x_plus = x.copy()
        x_plus[c] += change
        n0p = x_plus[0:3]
        n1p = x_plus[3:6]

        sp = get_strain_stretch_edge2D3D(n0p, n1p, l_k)
        grad_fdm[c] = (sp - strain0) / change

        grad_p, _ = grad_and_hess_strain_stretch_edge3D(n0p, n1p, l_k)
        hess_fdm[:,c] = (grad_p - grad) / change
    
    if iPlot:
        # compare
        plt.figure()
        plt.plot(grad,   'ro', label='analytic')
        plt.plot(grad_fdm,'b^',label='FD')
        plt.legend(); plt.xlabel('dof index'); plt.ylabel('∂ε/∂x'); plt.title('Gradient check')
        plt.show()
        
        plt.figure()
        plt.plot(hess.flatten(),    'ro', label='analytic')
        plt.plot(hess_fdm.flatten(), 'b^', label='FD')
        plt.legend(); plt.xlabel('entry index'); plt.ylabel('∂²ε/∂x²'); plt.title('Hessian check')
        plt.show()



# test_stretch_edge3D(iPlot)



# %% Loading mesh and get geometry variables ready.

# X0_4columns, ConnectivityMatrix_line, Triangles = load_mesh('mesh_python.txt')
# X0_4columns, ConnectivityMatrix_line, Triangles = load_mesh('mesh_python_27nodes.txt')
# X0_4columns, ConnectivityMatrix_line, Triangles = load_mesh('mesh_python_915nodes.txt')
# X0_4columns, ConnectivityMatrix_line, Triangles = load_mesh('mesh_python_beam.txt')


mesh_file = mesh_files[iMesh]
X0_4columns, ConnectivityMatrix_line, Triangles = load_mesh(mesh_file)

print("nodeXYZ:\n", X0_4columns)
print("Connectivity:\n", ConnectivityMatrix_line)
print("Triangles:\n", Triangles)

NP_total   = X0_4columns.shape[0]
Nedges  = ConnectivityMatrix_line.shape[0]
Ntriangles = Triangles.shape[0]

print(f"Number of nodes:     {NP_total}")
print(f"Number of edges:     {Nedges}")
print(f"Number of triangles: {Ntriangles}")


print("Ref edge lengths:")
L0 = fun_edge_lengths(X0_4columns, ConnectivityMatrix_line)
# for i, length in enumerate(L0):
    # print(f"L0_{i} = {length:.6f}")

# # convert to a 1D array of length N*3 with row‐major order.
X0 = X0_4columns[:,1:4].ravel()
print("vector X0 =", X0)

Ndofs=X0.shape
Ndofs=Ndofs[0]
print("Ndofs=", Ndofs)



# %% --2D-- plots for reference config

fig, ax = new_fig(1)
ax.triplot(
    X0_4columns[:, 1],        # x coords
    X0_4columns[:, 2],        # y coords
    (Triangles[:, 1:4]),      # zero‑based triangles
    color='blue'
)

ax.set_aspect('equal')
ax.set_title('Triangle Mesh')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()



if iPlot:

    # Fig. 2: nodes + triangle IDs
    fig, ax = new_fig(2)
    # draw triangles
    ax.triplot(
        X0_4columns[:, 1],
        X0_4columns[:, 2],
        (Triangles[:, 1:4]),
        color='blue'
    )
    
    # annotate triangle IDs
    for tri in Triangles:
        tid      = int(tri[0])
        idxs     = tri[1:4].astype(int)
        centroid = X0_4columns[idxs, 1:3].mean(axis=0)
        ax.text(
            centroid[0], centroid[1],
            str(tid),
            fontsize=12, color='red',
            ha='center', va='center'
        )
    
    # plot & label nodes
    for node in X0_4columns:
        nid = int(node[0])
        x, y = node[1], node[2]
        ax.scatter(x, y, color='pink', s=20)
        ax.text(
            x, y, str(nid),
            fontsize=12, color='black',
            ha='center', va='center'
        )
    
    ax.set_aspect('equal')
    ax.set_title('Nodes & Triangle IDs')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
    
    
    # plot nodes + edge IDs ---  
    fig, ax = new_fig(3)
    ax.set_aspect('equal')
    
    # draw edges
    for edge in ConnectivityMatrix_line:
        eid = int(edge[0])
        n1, n2 = int(edge[1]), int(edge[2])
        x0, y0 = X0_4columns[n1, 1], X0_4columns[n1, 2]
        x1, y1 = X0_4columns[n2, 1], X0_4columns[n2, 2]
        ax.plot([x0, x1], [y0, y1], color='gray', linewidth=1)
    
        # label edge at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(
            mx, my, str(eid),
            fontsize=15, color='blue',
            ha='center', va='center'
        )
    
    # plot & label nodes
    for node in X0_4columns:
        nid = int(node[0])
        x, y = node[1], node[2]
        ax.scatter(x, y, color='pink', s=20)
        ax.text(
            x, y, str(nid),
            fontsize=15, color='black',
            ha='center', va='center'
        )
    
    ax.set_title('Nodes & Edges IDs)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
    
    # plot nodes + edge IDs for a slender beam --- 
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_aspect('equal')
    
    # draw edges
    for eid, n1, n2 in ConnectivityMatrix_line.astype(int):
        x0, y0 = X0_4columns[n1,1], X0_4columns[n1,2]
        x1, y1 = X0_4columns[n2,1], X0_4columns[n2,2]
        ax.plot([x0, x1], [y0, y1], color='gray', linewidth=1)
        ax.text((x0+x1)/2, (y0+y1)/2, str(eid), fontsize=6,
                color='blue', ha='center', va='center')
    
    # draw nodes (fix unpacking)
    for node in X0_4columns:
        nid = int(node[0])
        x, y = node[1], node[2]
        ax.scatter(x, y, color='pink', s=10)
        ax.text(x, y, str(nid), fontsize=6,
                color='black', ha='center', va='center')
    
    # zoom to x in [0,1]
    ax.set_xlim(0, 1)
    ymin, ymax = X0_4columns[:,2].min(), X0_4columns[:,2].max()
    ax.set_ylim(ymin - 0.01, ymax + 0.01)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Nodes & Edges IDs: 0 ≤ x ≤ 1')
    plt.tight_layout()
    plt.show()
    



# %% --3D-- plots for reference config

# 3D triangle‐wireframe ---------
fig, ax = new_fig(4, projection='3d')

for tri in Triangles:
    # extract the 1‑based node IDs, convert to 0‑based indices
    i1, i2, i3 = int(tri[1]) , int(tri[2]) , int(tri[3]) 

    # grab their x,y,z from X0_4columns
    p0 = X0_4columns[i1, 1:4]
    p1 = X0_4columns[i2, 1:4]
    p2 = X0_4columns[i3, 1:4]

    # draw the triangle edges
    for a, b in ((p0, p1), (p1, p2), (p2, p0)):
        ax.plot([a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                color='blue', linewidth=0.5)

ax.set_title('3D Triangle Mesh')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
# auto‑scale
ax.auto_scale_xyz(X0_4columns[:,1],
                  X0_4columns[:,2],
                  X0_4columns[:,3])
plt.show()


if iPlot:
    
    # 3D nodes + triangle IDs -------
    fig, ax = new_fig(5, projection='3d')
    # lightly draw every triangle again
    for tri in Triangles:
        i1, i2, i3 = int(tri[1]), int(tri[2]), int(tri[3])
        p0 = X0_4columns[i1, 1:4]
        p1 = X0_4columns[i2, 1:4]
        p2 = X0_4columns[i3, 1:4]
        for a, b in ((p0, p1), (p1, p2), (p2, p0)):
            ax.plot([a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color='gray', linewidth=1)
    
    # annotate triangle IDs at centroids
    for tri in Triangles:
        tid = int(tri[0])
        idxs = [int(tri[j]) for j in (1,2,3)]
        pts = X0_4columns[idxs, 1:4]
        centroid = pts.mean(axis=0)
        ax.text(*centroid, str(tid), color='red')
    
    # plot & label nodes
    for node in X0_4columns:
        nid = int(node[0])
        x, y, z = node[1], node[2], node[3]
        ax.scatter(x, y, z, color='pink', s=20)
        ax.text(x, y, z, str(nid), color='black')
    
    ax.set_title('3D Nodes & Triangle IDs')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.auto_scale_xyz(X0_4columns[:,1],
                      X0_4columns[:,2],
                      X0_4columns[:,3])
    plt.show()
    
    
    # 3D nodes + edge IDs ---------
    fig, ax = new_fig(6, projection='3d')
    for edge in ConnectivityMatrix_line:
        eid = int(edge[0])
        i0, i1 = int(edge[1]), int(edge[2])
        p0 = X0_4columns[i0, 1:4]
        p1 = X0_4columns[i1, 1:4]
    
        # draw the edge
        ax.plot([p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color='gray', linewidth=1)
    
        # label at midpoint
        mid = (p0 + p1) / 2
        ax.text(*mid, str(eid), color='blue')
    
    # plot & label nodes
    for node in X0_4columns:
        nid = int(node[0])
        x, y, z = node[1], node[2], node[3]
        ax.scatter(x, y, z, color='pink', s=20)
        ax.text(x, y, z, str(nid), color='black')
    
    ax.set_title('3D Nodes & Edge IDs')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.auto_scale_xyz(X0_4columns[:,1],
                      X0_4columns[:,2],
                      X0_4columns[:,3])
    plt.show()




# %% Functions to assign thermal axial strains.


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


def assign_thermal_strains(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_thermal: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    inside: bool = True
) -> np.ndarray:
    """
    Assign thermal strain to each edge whose midpoint lies (inside/outside)
    the region defined by `region_fn`.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
        Coordinates of each node.
    connectivity : (Nedges,3) array
        Each row [eid, n0, n1].  We assume eid runs 0..Nedges‑1.
    eps_thermal : float
        Thermal strain to assign.
    region_fn : callable
        Given mids=(Nedges,3) array of midpoints, returns a boolean array
        of length Nedges: True where the edge should be “hot”.
    inside : bool
        If True, assign eps_thermal where region_fn is True;
        if False, assign where region_fn is False (i.e. outside).

    Returns
    -------
    epsilon_th : (Nedges,) array
        Thermal strain for each edge.
    """
    # 1) midpoints for every edge
    #    connectivity[:,1:3] are the two node‐indices per edge
    n0 = connectivity[:,1].astype(int)
    n1 = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # shape (Nedges,3)

    # 2) evaluate region test
    mask = region_fn(mids, node_xyz)  # boolean array length Nedges

    # 3) fill
    epsilon_th = np.zeros(connectivity.shape[0], dtype=float)
    if inside:
        epsilon_th[mask] = eps_thermal
    else:
        epsilon_th[~mask] = eps_thermal

    return epsilon_th



def assign_thermal_strains_contour(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_thermal: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    inside: bool = True
) -> np.ndarray:
    """
    Like assign_thermal_strains, but replaces the flat step
    by a smooth ramp that goes from 0 at the region boundary
    up to eps_thermal at the point farthest from that boundary
    (and vice‐versa if inside=False).

    Parameters
    ----------
    node_coords : (Nnodes,3) array
        nodal (x,y,z)
    connectivity : (Nedges,3) array
        each row [eid,n0,n1]
    eps_thermal : float
        peak thermal strain
    region_fn : callable
        given mids=(Nedges,3) returns a boolean mask
    inside : bool
        if True, ramp *inside* the mask; else ramp *outside*

    Returns
    -------
    epsilon_th : (Nedges,) array
    """
    # 1) midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) build mask
    mask = region_fn(mids)

    # 3) select the region we want to ramp over
    if inside:
        ramp_idx    = np.nonzero(mask)[0]
        boundary_idx = np.nonzero(~mask)[0]
    else:
        ramp_idx    = np.nonzero(~mask)[0]
        boundary_idx = np.nonzero(mask)[0]

    # 4) if either set is empty, fall back to flat
    if ramp_idx.size == 0 or boundary_idx.size == 0:
        flat = eps_thermal if inside else 0.0
        out  = np.full(mids.shape[0], flat, dtype=float)
        return out

    # 5) for each point in the ramp region, find its distance
    #    to the nearest point of the *other* region → d_i
    P_ramp     = mids[ramp_idx]     # (R,3)
    P_boundary = mids[boundary_idx] # (B,3)

    # compute pairwise squared‐distances R×B
    #    (this can handle a few thousand edges in a few seconds)
    diffs = P_ramp[:, None, :] - P_boundary[None, :, :]
    d2    = np.einsum('rbi,rbi->rb', diffs, diffs)
    d_min = np.sqrt(np.min(d2, axis=1))  # (R,)

    # 6) normalize so that the farthest ramp‐point has weight=1
    d_max = d_min.max()
    if d_max <= 0:
        # degenerate → flat
        weights = np.ones_like(d_min)
    else:
        weights = d_min / d_max

    # 7) build output
    epsilon_th = np.zeros(mids.shape[0], dtype=float)
    if inside:
        # inside the mask: 0 at the boundary, eps_thermal at farthest interior
        epsilon_th[ramp_idx] = eps_thermal * (1.0 - weights)
    else:
        # outside the mask: eps_thermal at the boundary, 0 at farthest outside
        epsilon_th[ramp_idx] = eps_thermal * weights

    return epsilon_th


def assign_thermal_strains_LinearTraisition(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_min: float,
    eps_max: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    inside: bool = True
) -> np.ndarray:
    """
    Assign thermal strain to each edge whose midpoint lies inside (or outside)
    the region defined by `region_fn`, but instead of a flat eps_thermal we
    do a linear ramp from eps_max at the mesh mid‐height line to eps_min at
    the top/bottom edges.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal (x,y,z)
    connectivity : (Nedges,3) array
      each row [eid, n0, n1]
    eps_min : float
      thermal strain at the top/bottom of the mesh (t=1)
    eps_max : float
      thermal strain on the mid‐height line (t=0)
    region_fn : callable
      given mids=(Nedges,3) array returns boolean mask
    inside : bool
      if True, apply ramp inside mask; else apply ramp outside
    """
    # 1) midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # shape (Nedges,3)

    # 2) region mask
    mask = region_fn(mids)

    # 3) compute normalized vertical dist t in [0,1]
    y_all  = node_coords[:,1]
    y_min  = y_all.min()
    y_max  = y_all.max()
    y_mid  = 0.5*(y_min + y_max)
    half_h = 0.5*(y_max - y_min)

    dy     = np.abs(mids[:,1] - y_mid)
    t_norm = np.clip(dy/half_h, 0.0, 1.0)

    # 4) allocate and fill
    epsilon_th = np.zeros(connectivity.shape[0], dtype=float)

    # interpolation formula: at t=0 → eps_min, at t=1 → eps_max
    ramp = eps_min + (eps_max - eps_min) * t_norm

    if inside:
        epsilon_th[mask]  = ramp[mask]
    else:
        epsilon_th[~mask] = ramp[~mask]

    return epsilon_th


def assign_thermal_strains_RadialTransition(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_min: float,
    eps_max: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    inside: bool = True
) -> np.ndarray:
    """
    Assign thermal strain to each edge whose midpoint lies inside (or outside)
    the region defined by `region_fn`, with a radial linear ramp from
    eps_max at the mesh center down to eps_min at the region boundary.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal (x,y,z)
    connectivity : (Nedges,3) array
      each row [eid, n0, n1]
    eps_min : float
      thermal strain at the farthest mask boundary (r = r_max)
    eps_max : float
      thermal strain at the mesh center (r = 0)
    region_fn : callable
      given mids=(Nedges,3) returns boolean mask
    inside : bool
      if True, apply ramp inside mask; else apply ramp outside
    """
    # 1) compute midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) build mask
    mask = region_fn(mids)   # boolean (Nedges,)

    # 3) compute mesh center in x–y
    x_all = node_coords[:,0]
    y_all = node_coords[:,1]
    x_mid = 0.5*(x_all.min() + x_all.max())
    y_mid = 0.5*(y_all.min() + y_all.max())

    # 4) compute radial distances for each midpoint
    dx    = mids[:,0] - x_mid
    dy    = mids[:,1] - y_mid
    r_all = np.sqrt(dx*dx + dy*dy)

    # 5) find maximum radius *within* the mask (so we ramp only to its boundary)
    if inside:
        r_max = r_all[mask].max() if np.any(mask) else 0.0
    else:
        r_max = r_all[~mask].max() if np.any(~mask) else 0.0

    # avoid division by zero
    if r_max <= 0:
        # all r==0 or empty region → flat eps_max or eps_min
        flat_value = eps_max if inside else eps_min
        return np.full(connectivity.shape[0], flat_value, dtype=float)

    # 6) normalized radius in [0,1]
    t_norm = np.clip(r_all / r_max, 0.0, 1.0)

    # 7) build ramp: at r=0 → eps_max; at r=r_max → eps_min
    ramp = eps_min + (eps_max - eps_min) * t_norm

    # 8) fill output
    epsilon_th = np.zeros(connectivity.shape[0], dtype=float)
    if inside:
        epsilon_th[mask]  = ramp[mask]
    else:
        epsilon_th[~mask] = ramp[~mask]

    return epsilon_th


def rect_region(mids: np.ndarray, node_xyz) -> np.ndarray:
    x, y = mids[:,0], mids[:,1]
    return (x >= x_min) & (x <= x_max) \
        & (y >= y_min) & (y <= y_max)


def circle_region(mids: np.ndarray,
                  node_xyz) -> np.ndarray:
    """
    True for edge‑midpoints within the circle of radius^2=r2 about (cx,cy).
    """
    x_coords = node_xyz[:, 0]
    y_coords = node_xyz[:, 1]
    cx = 0.5 * (x_coords.min() + x_coords.max())
    cy = 0.5 * (y_coords.min() + y_coords.max())
    radius = 0.5
    r2 = radius**2 

    dx = mids[:, 0] - cx
    dy = mids[:, 1] - cy
    return (dx*dx + dy*dy) <= r2




def bowl_region(mids: np.ndarray, node_xyz) -> np.ndarray:
    """
    Return True for edge‐midpoints in a central disk OR in every other
    wedge of the surrounding ring, producing a starburst pattern.
    """
    # 1) compute mesh center once
    x_coords = node_xyz[:, 0]
    y_coords = node_xyz[:, 1]
    cx = 0.5 * (x_coords.min() + x_coords.max())
    cy = 0.5 * (y_coords.min() + y_coords.max())

    # 2) radial limits
    r_inner = 0.5   # radius of the solid central disk
    r_outer = r_inner * 5.0  # outer edge of starburst ring

    # 3) how many wedges?
    n_wedges    = 12
    wedge_angle = 2*np.pi / n_wedges

    # 4) compute polar coords of each midpoint
    dx = mids[:,0] - cx
    dy = mids[:,1] - cy
    r   = np.sqrt(dx*dx + dy*dy)
    θ   = np.mod(np.arctan2(dy, dx), 2*np.pi)

    # 5) mask for central disk
    in_center = (r <= r_inner)

    # 6) mask for ring
    in_ring = (r > r_inner) & (r <= r_outer)

    # 7) alternating wedges: even‐numbered sectors only
    sector_idx   = (θ // wedge_angle).astype(int)
    in_even_wedge = (sector_idx % 2) == 0

    # 8) combine: center OR alternating wedges in ring
    return in_center | (in_ring & in_even_wedge)




# def circle_six_arms_region(
#     mids: np.ndarray,
#     *,
#     circle_center: Tuple[float,float] = (0.0, 0.0),
#     circle_radius: float      = Stripe_r,
#     arm_half_width: float     = StripeWidth,
#     arm_half_length: float    = StripeLength
# ) -> np.ndarray:
    
def circle_six_arms_region(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float],
    circle_radius: float,
    arm_half_width: float,
    arm_half_length: float
) -> np.ndarray:
    """
    Return True for edge‐midpoints lying inside:
      1) a small circle of radius `circle_radius` around `circle_center`, OR
      2) any of six rectangular "arms" of half-width `arm_half_width`
         and half-length `arm_half_length`, radiating at 0°, 60°, 120°, ….
    """
    x0, y0 = circle_center
    x, y   = mids[:,0] - x0, mids[:,1] - y0

    # 1) small circle mask
    in_circle = (x*x + y*y) <= circle_radius**2

    # 2) six arms at 60° increments
    arms = np.zeros_like(in_circle, dtype=bool)
    # angles for the arms
    phis = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for phi in phis:
        # unit vector along arm
        ux, uy = np.cos(phi), np.sin(phi)
        # projection onto the arm direction
        proj =  x*ux + y*uy
        # perpendicular distance from arm axis
        perp = np.abs(-x*uy + y*ux)
        # mask points within the rectangular arm
        arms |= (perp <= arm_half_width) & (proj >= -arm_half_length) & (proj <= arm_half_length)

    return in_circle | arms




# def square_X_region(
#     mids: np.ndarray,
#     *,
#     circle_center: Tuple[float,float] = Pattern_center,
#     circle_radius: float      = Stripe_r,
#     arm_half_width: float     = StripeWidth,
#     arm_half_length: float    = StripeLength
# ) -> np.ndarray:

    
def square_X_region(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float],
    circle_radius: float,
    arm_half_width: float,
    arm_half_length: float
) -> np.ndarray:
    """
    Return True for edge‐midpoints lying inside:
      1) a small circle of radius `circle_radius` around `circle_center`, OR
      2) two diagonal rectangular “arms” (an X shape) of half‐width
         `arm_half_width` and half‐length `arm_half_length` at ±45°.
    """
    # translate into local (X,Y) about the circle center
    cx, cy = circle_center
    X = mids[:,0] - cx
    Y = mids[:,1] - cy

    # 1) little central circle
    in_circle = (X*X + Y*Y) <= circle_radius**2

    # 2) two diagonal arms at φ=+45° and φ=−45°
    arms = np.zeros_like(in_circle, dtype=bool)
    for phi in (np.pi/4, -np.pi/4):
        ux, uy = np.cos(phi), np.sin(phi)
        # projection along the strip
        proj =  X*ux + Y*uy
        # signed distance perpendicular to the strip
        perp = np.abs(-X*uy + Y*ux)
        arms |= (
            (perp <= arm_half_width) &
            (proj >= -arm_half_length) &
            (proj <= +arm_half_length)
        )

    return in_circle | arms


    
# def stripe_region(
#     mids: np.ndarray,
#     node_xyz: np.ndarray = None,
#     *,
#     circle_center=Pattern_center, #(0.0, 0.0),
#     circle_radius=Stripe_r,
#     cross_half_width=StripeWidth,
#     cross_half_length=StripeLength
# ) -> np.ndarray:
    
def stripe_region(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float, float],
    circle_radius: float,
    cross_half_width: float,
    cross_half_length: float
) -> np.ndarray:
    
    x0, y0 = circle_center
    x, y   = mids[:,0], mids[:,1]

    # 1) small circle mask
    dx = x - x0
    dy = y - y0
    in_circle = (dx*dx + dy*dy) <= circle_radius**2

    # 2) horizontal bar mask: |y - y0| <= cross_half_width AND |x - x0| <= cross_half_length
    in_hbar = (np.abs(y - y0) <= cross_half_width) & (np.abs(x - x0) <= cross_half_length)

    # 3) vertical bar mask: |x - x0| <= cross_half_width AND |y - y0| <= cross_half_length
    in_vbar = (np.abs(x - x0) <= cross_half_width) & (np.abs(y - y0) <= cross_half_length)

    return  in_hbar
    # return in_circle | in_hbar | in_vbar


def assign_youngs_modulus(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    region_fn: Callable[[np.ndarray], np.ndarray],
    circle_center: Tuple[float,float],
    circle_radius: float,
    Ysoft: float,
    Yhard: float,
    Yratio: float,
    inside: bool = True
) -> np.ndarray:
    """
    Build a (Nedges,) array of Young's moduli:
      Ysoft inside the region, Yhard outside (or vice versa).
    """
    # 1) compute midpoints
    n0  = connectivity[:,1].astype(int)
    n1  = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])   # shape (Nedges,3)

    # 2) test region
    mask = region_fn(mids)   # boolean (Nedges,)

    # 3) piecewise assign *with* a radial ramp inside the region
    Y = np.empty(connectivity.shape[0], dtype=float)

    # --- compute radial distances from the circle center ---
    # You must know your circle_center=(cx,cy) and circle_radius=R
    cx, cy = circle_center
    R = circle_radius

    dx = mids[:,0] - cx
    dy = mids[:,1] - cy
    r  = np.sqrt(dx*dx + dy*dy)

    # normalized radius [0..1]
    r_norm = np.clip(r / R, 0.0, 1.0)

    # define the two soft‐moduli endpoints
    Ysoft_R  = Yratio * Ysoft    # at r = R
    Ysoft_r = Ysoft           # at r = 0
    Yhard_R  = Yratio * Yhard
    Yhard_r = Yhard

    if inside:
        Y[mask]  = Ysoft_r + (Ysoft_R - Ysoft_r) * r_norm[mask]
        # outside: hard material
        Y[~mask] = Yhard_r + (Yhard_R - Yhard_r) * r_norm[~mask]
    else:
        # inside: hard material
        Y[mask]  = Yhard_r + (Yhard_R - Yhard_r) * r_norm[mask]
        Y[~mask] = Ysoft_r + (Ysoft_R - Ysoft_r) * r_norm[~mask]

    return Y


def assign_youngs_modulus_v2(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    region_fn: Callable[[np.ndarray], np.ndarray],
    circle_center: Tuple[float,float],
    circle_radius: float,
    Ysoft: float,
    Yhard: float,
    Yratio: float,
    inside: bool = True,
    *,
    x_thresh: float = None,
    hard_factor: float = 1.0
) -> np.ndarray:
    """
    Build a (Nedges,) array of Young's moduli with two effects:
      1) a soft/hard radial ramp about circle_center (as before),
      2) anywhere x > x_thresh, force Y = hard_factor * Yhard_r.

    Parameters
    ----------
    node_coords : (Nnodes,3)
    connectivity: (Nedges,3)
    region_fn   : mids -> boolean (mask)
    circle_center: (cx,cy)
    circle_radius: R
    Ysoft, Yhard, Yratio: ramp endpoints
    inside      : apply inside‐mask ramp if True, else complement
    x_thresh    : if not None, x > x_thresh ⇒ override
    hard_factor : factor to multiply Yhard_r by in override zone
    """
    # 1) compute midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)
    x    = mids[:,0]

    # 2) mask & midpoint‐radius for ramp
    mask  = region_fn(mids)   # (Nedges,)
    cx, cy = circle_center
    dx = mids[:,0] - cx
    dy = mids[:,1] - cy
    r  = np.sqrt(dx*dx + dy*dy)
    r_norm = np.clip(r / circle_radius, 0.0, 1.0)

    # 3) soft/hard endpoints
    Ysoft_r = Ysoft
    Ysoft_R = Ysoft * Yratio
    Yhard_r = Yhard
    Yhard_R = Yhard * Yratio

    # 4) build the ramp arrays
    soft_ramp = Ysoft_r + (Ysoft_R - Ysoft_r) * r_norm
    hard_ramp = Yhard_r + (Yhard_R - Yhard_r) * r_norm

    # 5) allocate and fill
    Y = np.empty_like(r)
    if inside:
        Y[mask]  = soft_ramp[mask]
        Y[~mask] = hard_ramp[~mask]
    else:
        Y[mask]  = hard_ramp[mask]
        Y[~mask] = soft_ramp[~mask]

    # 6) override to super‐hard on the right side
    if x_thresh is not None:
        if inside:
            override = (~mask) & (x > x_thresh)
        else:
            override = (mask) & (x > x_thresh)

        Y[override] = hard_factor * Yhard_r

    return Y


def assign_youngs_modulus_v3(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    region_fn: Callable[[np.ndarray], np.ndarray],
    circle_center: Tuple[float,float],
    circle_radius: float,
    Ysoft: float,
    Yhard: float,
    Yratio: float,
    inside: bool = True,
    *,
    x_thresh_left: float = None,
    x_thresh_right: float = None,
    hard_factor: float = 1.0
) -> np.ndarray:
    """
    Build a (Nedges,) array of Young's moduli with two effects:
      1) a soft/hard radial ramp about circle_center (as before),
      2) anywhere x > x_thresh, force Y = hard_factor * Yhard_r.

    Parameters
    ----------
    node_coords : (Nnodes,3)
    connectivity: (Nedges,3)
    region_fn   : mids -> boolean (mask)
    circle_center: (cx,cy)
    circle_radius: R
    Ysoft, Yhard, Yratio: ramp endpoints
    inside      : apply inside‐mask ramp if True, else complement
    x_thresh    : if not None, x > x_thresh ⇒ override
    hard_factor : factor to multiply Yhard_r by in override zone
    """
    # 1) compute midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)
    x    = mids[:,0]

    # 2) mask & midpoint‐radius for ramp
    mask  = region_fn(mids)   # (Nedges,)
    cx, cy = circle_center
    dx = mids[:,0] - cx
    dy = mids[:,1] - cy
    r  = np.sqrt(dx*dx + dy*dy)
    r_norm = np.clip(r / circle_radius, 0.0, 1.0)

    # 3) soft/hard endpoints
    Ysoft_r = Ysoft
    Ysoft_R = Ysoft * Yratio
    Yhard_r = Yhard
    Yhard_R = Yhard * Yratio

    # 4) build the ramp arrays
    soft_ramp = Ysoft_r + (Ysoft_R - Ysoft_r) * r_norm
    hard_ramp = Yhard_r + (Yhard_R - Yhard_r) * r_norm

    # 5) allocate and fill
    Y = np.empty_like(r)
    if inside:
        Y[mask]  = soft_ramp[mask]
        Y[~mask] = hard_ramp[~mask]
    else:
        Y[mask]  = hard_ramp[mask]
        Y[~mask] = soft_ramp[~mask]

    # 6) override to super‐hard on the right side
    if x_thresh_left is not None:
        
        in_band = (x > x_thresh_left) & (x < x_thresh_right)
        if inside:
            override = (~mask) & in_band
        else:
            override = (mask) & in_band

        Y[override] = hard_factor * Yhard_r

    return Y


def two_half_rings_with_bar(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float] = (0.15, 0.08),
    separation: float = 0.01,
    inner_radius: float = 0.01,
    outer_radius: float = 0.05,
    bar_half_width: float = 0.002,
    wedge_angle: float = np.pi/6   # half‐angle of the missing wedge
) -> np.ndarray:
    """
    Return True for points inside two 'C'-shaped annuli (left+right)
    plus the rectangular bar joining their openings.

    The rings are full 360° annuli except for a wedge of angle 2*wedge_angle
    centered on the +x direction (for the left C) or -x direction (for the right C).
    """

    x0, y0 = circle_center
    # compute the two ring centers
    dx = separation/2
    xc_L, yc_L = x0 - dx, y0
    xc_R, yc_R = x0 + dx, y0

    x = mids[:,0]
    y = mids[:,1]

    # distances to centers
    dL = np.hypot(x - xc_L, y - yc_L)
    dR = np.hypot(x - xc_R, y - yc_R)

    # full annulus masks
    ann_L = (dL >= inner_radius) & (dL <= outer_radius)
    ann_R = (dR >= inner_radius) & (dR <= outer_radius)

    # compute angles
    thL = np.arctan2(y - yc_L, x - xc_L)   # in (-π, π]
    thR = np.arctan2(y - yc_R, x - xc_R)

    # for the left C, we remove the wedge around +x (th ≈ 0)
    keep_L = np.abs(thL) >= wedge_angle
    # but also only the “left” half‐plane x <= x0
    half_L = (x <= x0)

    # for the right C, remove the wedge around –x (th ≈ π or –π)
    # we shift angles to [–π, π] and remove |thR − π| < wedge_angle
    wrap_diff = np.mod(thR + np.pi, 2*np.pi) - np.pi  # shift into (–π, π]
    keep_R = np.abs(wrap_diff - np.pi) >= wedge_angle
    half_R = (x >= x0)

    mask_L = ann_L & keep_L & half_L
    mask_R = ann_R & keep_R & half_R

    # connecting bar between the two openings:
    #   y within +/- bar_half_width, x between the two ring centers
    bar_mask = (
        (y >= y0 - bar_half_width) &
        (y <= y0 + bar_half_width) &
        (x >= xc_L) &
        (x <= xc_R)
    )

    return mask_L | mask_R | bar_mask



def two_half_rings_with_spokes_and_bar(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float] = (0.15, 0.08),
    separation: float         = 0.10,
    inner_radius: float       = 0.04,
    outer_radius: float       = 0.08,
    arc_thickness: float      = 0.04,
    spoke_half_angle: float   = np.pi/180 * 3,  # 3° half-width
    bar_half_width: float     = 0.02
) -> np.ndarray:
    """
    Return True for points lying on the two 'C' shapes with:
      - inner & outer arcs
      - 6 radial spokes
      - connecting bar
    """
    x0, y0    = circle_center
    dx        = separation/2
    xc_L, yc_L = x0 - dx, y0
    xc_R, yc_R = x0 + dx, y0

    x = mids[:,0]
    y = mids[:,1]

    # 1) radial distances & basic annulus
    dL = np.hypot(x - xc_L, y - yc_L)
    dR = np.hypot(x - xc_R, y - yc_R)
    in_ann_L = (dL <= outer_radius)
    in_ann_R = (dR <= outer_radius)

    # 2) outer and inner arcs (thick), *with parentheses* around each comparison
    arc_L_outer = in_ann_L & (np.abs(dL - outer_radius) <= arc_thickness)
    arc_R_outer = in_ann_R & (np.abs(dR - outer_radius) <= arc_thickness)
    arc_L_inner = in_ann_L & (np.abs(dL - inner_radius) <= arc_thickness)
    arc_R_inner = in_ann_R & (np.abs(dR - inner_radius) <= arc_thickness)

    # restrict to half-annulus: left half for L, right half for R
    half_L = (x <= x0)
    half_R = (x >= x0)
    arc_L  = (arc_L_outer | arc_L_inner) & half_L
    arc_R  = (arc_R_outer | arc_R_inner) & half_R

    # 3) radial spokes (thin angular wedges), 6 spokes at 0°, 60°, …
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    thL    = np.arctan2(y - yc_L, x - xc_L)
    thR    = np.arctan2(y - yc_R, x - xc_R)

    def angular_diff(a, b):
        diff = (a - b + np.pi) % (2*np.pi) - np.pi
        return diff

    spoke_L = np.zeros_like(x, dtype=bool)
    spoke_R = np.zeros_like(x, dtype=bool)
    for phi in angles:
        spoke_L |= (
            in_ann_L & half_L &
            (np.abs(angular_diff(thL, phi)) <= spoke_half_angle)
        )
        spoke_R |= (
            in_ann_R & half_R &
            (np.abs(angular_diff(thR, phi)) <= spoke_half_angle)
        )

    # 4) connecting bar
    bar_mask = (
        (y >= y0 - bar_half_width) &
        (y <= y0 + bar_half_width) &
        (x >= xc_L) &
        (x <= xc_R)
    )

    return arc_L | arc_R | spoke_L | spoke_R | bar_mask


def add_boundary_fluctuations(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    epsilon_th: np.ndarray,
    amp: float = 1e-3,
    n_waves: int = 4,
    decay_width: float = 0.05
) -> np.ndarray:
    """
    Add a small fluctuation to epsilon_th that is largest at the four
    edges of the rectangular domain and decays into the interior.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal coordinates
    connectivity : (Nedges,3) array
      edge list [eid, n0, n1]
    epsilon_th : (Nedges,) array
      existing thermal strains
    amp : float
      maximum fluctuation amplitude at the boundary
    n_waves : int
      how many full sin‐cycles along each edge
    decay_width : float
      distance over which the fluctuation decays to zero inwards
    """
    # 1) get midpoints back
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) domain extents
    x = mids[:,0]
    y = mids[:,1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    Lx = xmax - xmin
    Ly = ymax - ymin

    # 3) distance to each of the four edges
    d_left   = x - xmin
    d_right  = xmax - x
    d_bottom = y - ymin
    d_top    = ymax - y

    # 4) choose the closest boundary for each midpoint
    d_edge = np.minimum.reduce([d_left, d_right, d_bottom, d_top])

    # 5) build a decaying envelope exp(−d_edge/decay_width)
    envelope = np.exp(-d_edge / decay_width)

    # 6) build a sinusoidal fluctuation along the boundary coordinate:
    #    we’ll project each point onto its closest edge, then
    #    parameterize that edge by a coordinate s in [0,1].
    #    For simplicity we’ll use x/Lx for top/bottom and y/Ly for left/right.
    #    (This mixes a bit, but gives four “bands” of waves.)
    s = np.zeros_like(x)
    # where the closest is left or right, use y
    mask_v = (d_edge == d_left) | (d_edge == d_right)
    s[mask_v] = (y[mask_v] - ymin) / Ly
    # where the closest is top or bottom, use x
    mask_h = ~mask_v
    s[mask_h] = (x[mask_h] - xmin) / Lx

    # 7) fluctuation = amp * envelope * sin(2π * n_waves * s)
    fluct = amp * envelope * np.sin(2*np.pi*n_waves*s)

    # 8) return the superposition
    return epsilon_th + fluct


def add_OneSide_boundary_fluctuations(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    epsilon_th: np.ndarray,
    amp: float = 1e-3,
    n_waves: int = 4,
    decay_width: float = 0.05
) -> np.ndarray:
    """
    Add a small fluctuation to epsilon_th that is largest at the four
    edges of the rectangular domain and decays into the interior.

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal coordinates
    connectivity : (Nedges,3) array
      edge list [eid, n0, n1]
    epsilon_th : (Nedges,) array
      existing thermal strains
    amp : float
      maximum fluctuation amplitude at the boundary
    n_waves : int
      how many full sin‐cycles along each edge
    decay_width : float
      distance over which the fluctuation decays to zero inwards
    """
    # 1) get midpoints back
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) domain extents
    x = mids[:,0]
    y = mids[:,1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    Lx = xmax - xmin
    Ly = ymax - ymin

    # 3) distance to each of the four edges
    d_left   = x - xmin
    d_right  = xmax - x
    d_bottom = y - ymin
    d_top    = ymax - y

    # 4) choose the closest boundary for each midpoint
    d_edge = np.minimum.reduce([d_left])

    # 5) build a decaying envelope exp(−d_edge/decay_width)
    envelope = np.exp(-d_edge / decay_width)

    # 6) build a sinusoidal fluctuation along the boundary coordinate:
    #    we’ll project each point onto its closest edge, then
    #    parameterize that edge by a coordinate s in [0,1].
    #    For simplicity we’ll use x/Lx for top/bottom and y/Ly for left/right.
    #    (This mixes a bit, but gives four “bands” of waves.)
    s = np.zeros_like(x)
    # where the closest is left or right, use y
    mask_v = (d_edge == d_left) | (d_edge == d_right)
    s[mask_v] = (y[mask_v] - ymin) / Ly
    # where the closest is top or bottom, use x
    mask_h = ~mask_v
    s[mask_h] = (x[mask_h] - xmin) / Lx

    # 7) fluctuation = amp * envelope * sin(2π * n_waves * s)
    fluct = amp * envelope * np.sin(2*np.pi*n_waves*s)

    # 8) return the superposition
    return epsilon_th + fluct


def white_C_pattern_mask(
    mids: np.ndarray,
    *,
    circle_center: Tuple[float,float],
    separation: float,
    inner_radius: float,
    outer_radius: float,
    arc_thickness: float,
    n_spokes: int = 6,
    spoke_half_angle: float = np.deg2rad(3),
    bar_half_width: float
) -> np.ndarray:
    """
    Return a boolean mask of shape (Nedges,) that is True wherever
    an edge‐midpoint falls on the white 'C' pattern:
      - two half‐annuli (inner & outer arcs of thickness `arc_thickness`)
      - n_spokes radial spokes
      - a straight bar connecting the two openings.

    Parameters
    ----------
    mids : (Nedges,3) array of edge midpoints
    circle_center : (x0,y0) centerline of the two C’s
    separation    : distance between the two C‐centers along x
    inner_radius  : inner radius of each C
    outer_radius  : outer radius of each C
    arc_thickness : radial thickness of each arc (≈ line‐width/2)
    n_spokes      : how many spokes per C (default 6)
    spoke_half_angle : half‐angular width of each spoke (radians)
    bar_half_width   : half‐height of the joining bar
    """
    x0,y0 = circle_center
    dx    = separation/2
    # left‐ and right‐centers
    centers = [(x0-dx,y0), (x0+dx,y0)]
    x_mid,y_mid = x0,y0

    x = mids[:,0]
    y = mids[:,1]
    mask = np.zeros(len(mids), bool)

    # Build each C
    for (xc,yc) in centers:
        dx_ = x - xc
        dy_ = y - yc
        r  = np.hypot(dx_, dy_)

        # 1) thick arcs (inner & outer)
        arc_outer = (np.abs(r-outer_radius) <= arc_thickness)
        arc_inner = (np.abs(r-inner_radius) <= arc_thickness)
        # restrict to the correct half‐plane
        if xc < x_mid:
            halfplane = (x <= x_mid)
        else:
            halfplane = (x >= x_mid)
        mask |= (arc_outer | arc_inner) & halfplane

        # 2) radial spokes
        th = np.arctan2(dy_, dx_)   # in (−π,π]
        for k in range(n_spokes):
            phi = 2*np.pi * k / n_spokes
            diff = (th - phi + np.pi) % (2*np.pi) - np.pi
            spoke = (np.abs(diff) <= spoke_half_angle) \
                    & (r >= inner_radius) & (r <= outer_radius) \
                    & halfplane
            mask |= spoke

    # 3) straight bar between the two C‐openings
    xcL,ycL = centers[0]
    xcR,ycR = centers[1]
    bar = (
        (np.abs(y - y_mid) <= bar_half_width)
        & (x >= xcL) & (x <= xcR)
    )
    mask |= bar

    return mask



def white_star_pattern_mask(
    mids: np.ndarray,
    *,
    center: Tuple[float,float] = (0.2, 0.1),
    star_radius: float    = 0.05,
    star_thickness: float = 0.002,
    n_spokes: int         = 6,
    bar_length: float     = 0.15,
    bar_thickness: float  = 0.002
) -> np.ndarray:
    """
    Return a boolean mask for points lying on:
      - a 'star' of n_spokes equally spaced (60° apart if n_spokes=6),
        each spoke of length star_radius and thickness star_thickness,
      - plus a straight bar extending to the right of the center
        of length bar_length and thickeness bar_thickness.

    mids : (Nedges,3) array of edge‐midpoints
    center : (cx,cy) center of the star
    star_radius : how far each of the 6 spokes reaches
    star_thickness : half‐width of each spoke
    bar_length : length of the extra rightward arm
    bar_thickness : half‐width of that bar
    """
    cx, cy = center
    # translate to star‐centered coords
    X = mids[:,0] - cx
    Y = mids[:,1] - cy

    mask = np.zeros_like(X, dtype=bool)

    # 1) the six star‐spokes, every 360/n_spokes degrees
    phis = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)
    for phi in phis:
        ux, uy = np.cos(phi), np.sin(phi)
        # projection along the spoke:
        proj =  X*ux + Y*uy
        # perpendicular distance to the spoke‐axis:
        perp = np.abs(-X*uy + Y*ux)
        mask |= (proj >= 0) & (proj <= star_radius) & (perp <= star_thickness)

    # 2) the extra long rightward bar (phi=0)
    proj_bar = X           # since phi=0 ⇒ u=(1,0) ⇒ proj = X
    perp_bar = np.abs(Y)   # perpendicular distance is just |y|
    mask |= (proj_bar >= 0) & (proj_bar <= bar_length) & (perp_bar <= bar_thickness)

    return mask



def whole_peanut_pattern_mask(
    mids: np.ndarray,
    *,
    left_center:  Tuple[float,float],
    right_center: Tuple[float,float],
    star_radius:  float   = 0.05,
    star_thickness: float = 0.002,
    n_spokes:      int    = 6,
    beam_thickness: float = 0.002
) -> np.ndarray:
    """
    Return True for edge-midpoints lying on:
      - two 'star' shapes at left_center and right_center,
        each with n_spokes radial arms of length star_radius
        and half-width star_thickness;
      - plus a rectangular beam connecting the two centers,
        of half-width beam_thickness.

    mids            : (Nedges,3) array of midpoints
    left_center     : (xL,yL)
    right_center    : (xR,yR)
    star_radius     : radial length of each spoke
    star_thickness  : half-width of each spoke arm
    n_spokes        : number of arms per star
    beam_thickness  : half-width of the connecting beam
    """
    xL,yL = left_center
    xR,yR = right_center
    X = mids[:,0]
    Y = mids[:,1]
    mask = np.zeros_like(X, dtype=bool)

    # ---- helper to build one star at (cx,cy) ----
    def _star_mask(cx, cy):
        # translate to local coords
        dx = X - cx
        dy = Y - cy
        local = np.zeros_like(dx, dtype=bool)

        phis = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)
        for phi in phis:
            ux, uy = np.cos(phi), np.sin(phi)
            proj =  dx*ux + dy*uy
            perp = np.abs(-dx*uy + dy*ux)
            local |= (proj >= 0) & (proj <= star_radius) & (perp <= star_thickness)

        return local

    # draw left and right stars
    mask |= _star_mask(xL,yL)
    mask |= _star_mask(xR,yR)

    # ---- beam between the two star centers ----
    # restrict X between xL and xR (in either order)
    x_min, x_max = min(xL, xR), max(xL, xR)
    beam = (
        (X >= x_min) & (X <= x_max) &
        (np.abs(Y - ((yL+yR)/2)) <= beam_thickness)
    )
    mask |= beam

    return mask


def whole_peanut_region(
    mids: np.ndarray,
    *,
    center: Tuple[float, float],
    delta_shape: float      = 0.05,
    star_radius: float      = 0.05,
    star_thickness: float   = 0.002,
    n_spokes: int           = 6,
    beam_thickness: float   = 0.002
) -> np.ndarray:
    """
    Region‐mask for the “whole‑peanut” pattern:
      • two star‑shapes at (center_x ± delta_shape, center_y)
      • joined by a rectangular beam of half‑width beam_thickness.

    Parameters
    ----------
    mids : (Nedges,3) array  
        coordinates of edge‑midpoints
    center : (x_center, y_center)  
        midpoint between the two star centers
    delta_shape : float  
        half the distance between the star centers along x
    star_radius : float  
        length of each star’s radial arm
    star_thickness : float  
        half‑width of each arm
    n_spokes : int  
        number of arms per star
    beam_thickness : float  
        half‑width of the beam connecting the stars

    Returns
    -------
    mask : (Nedges,) boolean array  
        True where mids lie in the peanut region
    """
    x0, y0 = center
    left_center  = (x0 - delta_shape, y0)
    right_center = (x0 + delta_shape, y0)

    return whole_peanut_pattern_mask(
        mids,
        left_center   = left_center,
        right_center  = right_center,
        star_radius   = star_radius,
        star_thickness= star_thickness,
        n_spokes      = n_spokes,
        beam_thickness= beam_thickness
    )



def assign_thermal_strains_EllipticTransition(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    eps_min: float,
    eps_max: float,
    region_fn: Callable[[np.ndarray], np.ndarray],
    ellipse_axes: Tuple[float,float] = (1.0,1.0),
    inside: bool = True
) -> np.ndarray:
    """
    Same as your radial ramp, but distance is measured in an ellipse:
      ((x - x_mid)/a)^2 + ((y - y_mid)/b)^2 = 1

    Parameters
    ----------
    node_coords : (Nnodes,3) array
      nodal (x,y,z)
    connectivity : (Nedges,3) array
      each row [eid, n0, n1]
    eps_min : float
      thermal strain at the ellipse boundary (t_norm = 1)
    eps_max : float
      thermal strain at the center      (t_norm = 0)
    region_fn : callable
      given mids=(Nedges,3) returns boolean mask
    ellipse_axes : (a,b)
      semi‐axes of the ellipse in x‐ and y‐directions
    inside : bool
      if True, apply ramp inside mask; else apply ramp outside
    """
    # 1) edge midpoints
    n0   = connectivity[:,1].astype(int)
    n1   = connectivity[:,2].astype(int)
    mids = 0.5*(node_coords[n0] + node_coords[n1])  # (Nedges,3)

    # 2) boolean mask from user
    mask = region_fn(mids)   # (Nedges,)

    # 3) compute global mesh center
    x_all = node_coords[:,0]
    y_all = node_coords[:,1]
    x_mid = 0.5*(x_all.min() + x_all.max())
    y_mid = 0.5*(y_all.min() + y_all.max())

    # 4) compute normalized elliptical radius
    a, b = ellipse_axes
    dx   = mids[:,0] - x_mid
    dy   = mids[:,1] - y_mid
    # ellipse “radius” r_ell in [0..∞):
    r_ell = np.sqrt((dx/a)**2 + (dy/b)**2)

    # 5) pick the max ellipse‐radius *within* the mask (or its complement)
    if inside:
        r_max = r_ell[mask].max()  if np.any(mask)  else 0.0
    else:
        r_max = r_ell[~mask].max() if np.any(~mask) else 0.0

    # 6) handle degenerate case
    if r_max <= 0:
        flat = eps_max if inside else eps_min
        return np.full(connectivity.shape[0], flat, dtype=float)

    # 7) normalize to [0..1], build linear ramp eps=eps_max→eps_min
    t_norm = np.clip(r_ell / r_max, 0.0, 1.0)
    ramp   = eps_min + (eps_max - eps_min)*t_norm

    # 8) fill only the mask (or its complement)
    epsilon_th = np.zeros(connectivity.shape[0], dtype=float)
    if inside:
        epsilon_th[mask]   = ramp[mask]
    else:
        epsilon_th[~mask]  = ramp[~mask]

    return epsilon_th


node_xyz = X0_4columns[:,1:4]   # shape (Nnodes,3)
xs = node_xyz[:,0]
ys = node_xyz[:,1]

x_center = 0.5*(xs.min() + xs.max())
y_center = 0.5*(ys.min() + ys.max())

print(f"Domain center: x={x_center:.4f}, y={y_center:.4f}")


ellipse_axes = ((xs.max()-xs.min())/2,
                (ys.max()-ys.min())/2)


Pattern_center=(x_center, y_center)



if iMesh == 2:
    region_fn = partial(
        whole_peanut_region,
        center         = Pattern_center,
        delta_shape    = delta_shape,
        star_radius    = star_radius,
        star_thickness = star_thickness,
        n_spokes       = n_spokes,
        beam_thickness = beam_thickness
    )

elif iMesh in (1, 3):

    if iMesh == 1:
        region_fn = partial(
            circle_six_arms_region,
            circle_center   = Pattern_center,
            circle_radius   = Stripe_r,
            arm_half_width  = StripeWidth,
            arm_half_length = StripeLength
        )
    else:  # iMesh == 3
        region_fn = partial(
            square_X_region,
            circle_center   = Pattern_center,
            circle_radius   = Stripe_r,
            arm_half_width  = StripeWidth,
            arm_half_length = StripeLength
        )

else:
    raise ValueError(f"Unsupported mesh index: {iMesh!r}")


if iMesh in (1, 3):

    eps_th_vector = assign_thermal_strains_contour(
        node_xyz,
        ConnectivityMatrix_line,
        eps_thermal,
        region_fn=region_fn,
        inside=False,
    )

    Y_array = assign_youngs_modulus(
        node_xyz,
        ConnectivityMatrix_line,
        region_fn=region_fn,
        circle_center=Pattern_center,
        circle_radius=OuterR,
        Ysoft=Ysoft,
        Yhard=Yhard,
        Yratio=Yratio,
        inside=False,
    )

elif iMesh == 2:

    eps_th_vector = assign_thermal_strains_contour(
        node_xyz,
        ConnectivityMatrix_line,
        eps_thermal,
        region_fn=region_fn,
        inside=False,
    )

    if iFluc == 1:
        eps_th_vector = add_boundary_fluctuations(
            node_xyz,
            ConnectivityMatrix_line,
            eps_th_vector,
            amp         = eps_thermal*5,   # max fluctuation
            n_waves     = 0.5,             # sin‑cycles along each side
            decay_width = 0.03,            # fade‑into‑interior width
        )

    Y_array = assign_youngs_modulus_v3(
        node_xyz,
        ConnectivityMatrix_line,
        region_fn     = region_fn,
        circle_center = Pattern_center,
        circle_radius = OuterR,
        Ysoft         = Ysoft,
        Yhard         = Yhard,
        Yratio        = Yratio,
        inside        = False,
        x_thresh_left =  0.10,
        x_thresh_right=  0.185,
        hard_factor   =  1.0,
    )

else:
    raise ValueError(f"Unsupported mesh index for thermal/Y assignment: {iMesh!r}")


# if iMesh in (1, 2, 3, 5, 6, 7, 8, 9):
    
#     if iPatterns == 1:

#         eps_th_vector = assign_thermal_strains_contour(
#             node_xyz,
#             ConnectivityMatrix_line,
#             eps_thermal,
#             region_fn,
#             inside=False)
        
#         Y_array = assign_youngs_modulus(
#             node_xyz,
#             ConnectivityMatrix_line,
#             region_fn,
#             circle_center=Pattern_center,
#             circle_radius=OuterR,
#             Ysoft=Ysoft,
#             Yhard=Yhard,
#             Yratio=Yratio,
#             inside=False)

#     if iPatterns == 2:
                           
#         eps_th_vector = assign_thermal_strains_contour(
#             node_xyz,
#             ConnectivityMatrix_line,
#             eps_thermal,
#             region_fn,
#             inside=False)
        
#         ## Superpose fluctuated thermal strains
#         if iFluc == 1:
#                 eps_th_vector = add_boundary_fluctuations(
#                     node_xyz,
#                     ConnectivityMatrix_line,
#                     eps_th_vector,
#                     amp           = eps_thermal*5, # max fluctuation
#                     n_waves       = 0.5,               # number of sin-cycles along each side
#                     decay_width   = 0.03)            # how quickly it fades into the interior

#         ## Two Y values within one region, two regions
#         Y_array = assign_youngs_modulus_v3(
#             node_xyz,
#             ConnectivityMatrix_line,
#             region_fn=region_fn,
#             circle_center=Pattern_center,
#             circle_radius=OuterR,
#             Ysoft=Ysoft,
#             Yhard=Yhard,
#             Yratio=Yratio,
#             inside=False,
#             x_thresh_left  = 0.10,
#             x_thresh_right = 0.185,
#             hard_factor = 1.0)     




plot_thermal_strain_edges(
    node_coords = node_xyz,
    connectivity= ConnectivityMatrix_line,
    epsilon_th  = eps_th_vector,
    title       = "thermal axial strain",
    cmap        = 'coolwarm'  # or viridis, plasma, etc.
)


    
plot_thermal_strain_edges(
    node_coords = node_xyz,
    connectivity= ConnectivityMatrix_line,
    epsilon_th  = Y_array,
    title       = "Young's modulus",
    cmap        = 'coolwarm'
)


plot_thermal_strain_edges_CustomRange(
    node_coords = node_xyz,
    connectivity= ConnectivityMatrix_line,
    epsilon_th  = Y_array,
    title       = "Young's modulus (by Yhard)",
    cmap        = 'coolwarm',    
    vmin        = Yratio*Yhard,
    vmax        = Yhard)

plot_thermal_strain_edges_CustomRange(
    node_coords = node_xyz,
    connectivity= ConnectivityMatrix_line,
    epsilon_th  = Y_array,
    title       = "Young's modulus (by Ysoft)",
    cmap        = 'coolwarm',    
    vmin        = Yratio*Ysoft,
    vmax        = Ysoft)






# Y_array = np.where(
#     eps_th_vector == 0.00,   # mask
#     Yhard,                      # if eps_th == 0
#     np.where(
#       eps_th_vector == eps_thermal, # else if eps_th == 0.01
#       Ysoft,
#       Ysoft                     # fallback if neither exactly 0 nor 0.01
#     )
# )

# ks_array = Y_array * h * (lk**2) * np.sqrt(3) / 2.0


# 07-08-25
ks_array = np.where(Y_array == Yhard, ks12,
           np.where(Y_array == Ysoft, ks1, 0.0))


# kb is defined after hinge_quad_order defined.
# kb = Ysoft * (h**3) / ( np.sqrt(3) * 6.0 )




# Have eps_th_vector. To do:
# 1. Modify stretch part: fun_grad_hess_energy_stretch_linear_elastic_edge. Done.
#                         
# 2. Modify bend part: fun_coupled_Ebend_grad_hess. Done.
#                      
# 3. Modify Total energy: fun_total_system_energy_coupled. Done.



# %% Def. functions: gradient, Hessian of stretch energy: dEs/dx, d^2(Es)/dx^2.
#  tidy up after 1:52pm.

#  Replace class ElasticForce with a class ElasticGHEdges. Done
# 1. Modify ElasticForce.computeGradientHessian to loop with edge IDs. Done
# 2. Modify grad_and_hess_energy_stretch_linear_elastic to use G, H directly
#    without averaging. Replace "grad_and_hess_energy_stretch_linear_elastic" with "fun_grad_hess_energy_stretch_linear_elastic_edge" Done
#    Replace nodal "get_strain_stretch2D" with elemental "get_strain_stretch_edge2D", Done
#    Replace nodal "grad_and_hess_strain_stretch2D" with elemental "grad_and_hess_strain_stretch_edge2D" Done


def fun_grad_hess_energy_stretch_linear_elastic_edge(node0, node1, l_0 = None, ks = None):
    # H has two terms, material elastic stiffness + geometric stiffness
    
    strain_stretch = get_strain_stretch_edge2D3D(node0, node1, l_0)
    G_strain, H_strain = grad_and_hess_strain_stretch_edge3D(node0, node1, l_0)

    gradE_strain = ks * strain_stretch * l_0
    hessE_strain = ks * l_0

    G = gradE_strain * G_strain
    H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)
    
    sub_block = H[0:3, 0:3]
    squared = sub_block**2
    sum_of_squares = np.sum(squared)
    # Verify stiffness with norm in small strain. With finite strain, geometric stiffness matters.
    print("Spring stiffness verify, node1=",node1)
    print("Sum of squares for 3DOFs from H:", sum_of_squares)
    print("squared ks/l0 :", (ks/l_0)**2)

    return G, H
    # The conpoment of H=ks/l0, verify the stretch spring stiffness.
    # np.outer(G_strain, G_strain) orients the stiffness.



def fun_grad_hess_energy_stretch_linear_elastic_edge_thermal(node0, node1, l_0 = None, ks = None, eps_th=0.0):
    # H has two terms, material elastic stiffness + geometric stiffness
    
    strain_stretch = get_strain_stretch_edge2D3D(node0, node1, l_0) - eps_th
    G_strain, H_strain = grad_and_hess_strain_stretch_edge3D(node0, node1, l_0)

    gradE_strain = ks * strain_stretch * l_0
    hessE_strain = ks * l_0

    G = gradE_strain * G_strain
    H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)
    
    sub_block = H[0:3, 0:3]
    squared = sub_block**2
    sum_of_squares = np.sum(squared)
    # # Verify stiffness with norm in small strain. With finite strain, geometric stiffness matters.
    # print("Spring stiffness verify, node1=",node1)
    # print("Sum of squares for 3DOFs from H:", sum_of_squares)
    # print("squared ks/l0 :", (ks/l_0)**2)

    return G, H
    # The conpoment of H=ks/l0, verify the stretch spring stiffness.
    # np.outer(G_strain, G_strain) orients the stiffness.



# %% Def. functions Dihedral angle.

def signedAngle(u = None,v = None,n = None):
    # This function calculates the signed angle between two vectors, "u" and "v",
    # using an optional axis vector "n" to determine the direction of the angle.
    #
    # Parameters:
    #   u: numpy array-like, shape (3,), the first vector.
    #   v: numpy array-like, shape (3,), the second vector.
    #   n: numpy array-like, shape (3,), the axis vector that defines the plane
    #      in which the angle is measured. It determines the sign of the angle.
    #
    # Returns:
    #   angle: float, the signed angle (in radians) from vector "u" to vector "v".
    #          The angle is positive if the rotation from "u" to "v" follows
    #          the right-hand rule with respect to the axis "n", and negative otherwise.
    #
    # The function works by:
    # 1. Computing the cross product "w" of "u" and "v" to find the vector orthogonal
    #    to both "u" and "v".
    # 2. Calculating the angle between "u" and "v" using the arctan2 function, which
    #    returns the angle based on the norm of "w" (magnitude of the cross product)
    #    and the dot product of "u" and "v".
    # 3. Using the dot product of "n" and "w" to determine the sign of the angle.
    #    If this dot product is negative, the angle is adjusted to be negative.
    #
    # Example:
    #   signedAngle(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    #   This would return a positive angle (π/2 radians), as the rotation
    #   from the x-axis to the y-axis is counterclockwise when viewed along the z-axis.
    w = np.cross(u,v)
    angle = np.arctan2( np.linalg.norm(w), np.dot(u,v) )
    if (np.dot(n,w) < 0):
        angle = - angle

    return angle


def mmt(matrix):
    return matrix + matrix.T

#          x2
#          /\
#         /  \
#      e1/    \e3
#       /  t0  \
#      /        \
#     /A1  e0  A3\
#   x0------------x1
#     \A2      A4/
#      \   t1   /
#       \      /
#      e2\    /e4
#         \  /
#          \/
#          x3
#
#  Edge orientation: e0,e1,e2 point away from x0
#                       e3,e4 point away from x1


def getTheta(x0, x1 = None, x2 = None, x3 = None):

    if np.size(x0) == 12:  # Allow another type of input where x0 contains all the info
      x1 = x0[3:6]
      x2 = x0[6:9]
      x3 = x0[9:12]
      x0 = x0[0:3]

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0

    n0 = np.cross(m_e0, m_e1)
    n1 = np.cross(m_e2, m_e0)

    # Calculate the signed angle using the provided function
    theta = signedAngle(n0, n1, m_e0)

    return theta

# In the original code, there are probaly TWO sign errors in the expressions for m_h3 and m_h4.
# [Original code: % https://github.com/shift09/plates-shells/blob/master/src/bending.cpp]
# I indicated those two corrections by writing the word "CORRECTION" next
# to them.

# %% Test dihedral angles, get HingeQuads.

# Completed: 
    # 1 Get hinge IDs from the mesh. Done
    # 2 Get hinge edge ID with corresponding 4 node IDs. Done
    #   Try see if there are more efficient way. Not much needed. Done
    
    # 3 For each hinge pair, the randomly determine x0, x1.
    #   For x2, x3, if the two normals u, v are towards to z positive direction, keep the order.
    #   If dot with z axis normal <0, swap x2 and x3.
    #   Then save the order matrix: hinge edge ID, x0, x1, x2, x3. Done
    # 4 Add options to use stretch, bending, or both in energy. Done
    # 5 Calculate grad, hess and assemble them. Done
    # 6 Compare Euler-Bernouli beam deflection with (pure) bending case? Done, verified.



from collections import Counter

# assume Triangles (n_tris×4) and ConnectivityMatrix_line (n_edges×3) are in scope

# 1) Build a flat list of all undirected edges from the triangle list
tri_edges = []
for _, v1, v2, v3 in Triangles.astype(int):
    tri_edges += [
        tuple(sorted((v1, v2))),
        tuple(sorted((v2, v3))),
        tuple(sorted((v3, v1))),
    ]

# 2) Count how many times each edge appears
edge_counts = Counter(tri_edges)

# 3) Keep only those edges that appear exactly twice (interior edges = hinges)
hinge_keys = {edge for edge, cnt in edge_counts.items() if cnt == 2}

# 4) Extract their IDs from your connectivity table
hinge_edges = []
for eid, n0, n1 in ConnectivityMatrix_line.astype(int):
    if tuple(sorted((n0, n1))) in hinge_keys:
        hinge_edges.append(eid)

hinge_edges = np.array(hinge_edges, dtype=int)
print("Hinge edge IDs:", hinge_edges)


all_edges = np.arange(Nedges, dtype=int)

# 2) boundary edges are those in all_edges but not in hinge_edges
boundary_edges = np.setdiff1d(all_edges, hinge_edges)

print("All edges:     ", all_edges)
print("Hinge edges:   ", hinge_edges)
print("Boundary edges:", boundary_edges)



# Triangles           : array of shape (n_tris, 4) with [triID, v1, v2, v3]
# ConnectivityMatrix_line : array of shape (n_edges, 3) with [edgeID, n0, n1]
# hinge_edges         : 1D array of edgeIDs that lie in the interior

# --- Pass 1: build edge → list of opposite‐vertices ---
edge_to_opps = defaultdict(list)

for _, v1, v2, v3 in Triangles.astype(int):
    verts = (v1, v2, v3)
    # for each of the 3 edges of this triangle:
    for a, b in combinations(verts, 2):
        key = tuple(sorted((a, b)))      # undirected edge
        # the “opposite” vertex is the one not in (a,b)
        opp = next(v for v in verts if v not in key)
        edge_to_opps[key].append(opp)

# Now any interior edge (shared by two triangles) has exactly two opposites:
#   len(edge_to_opps[key]) == 2

# --- Pass 2: collect the four nodes for each hinge edgeID ---
hinge_quads = []  # will hold rows [edgeID, n0, n1, oppA, oppB]

# build a quick lookup from edgeID -> (n0, n1)
edgeid_to_nodes = {int(eid): (int(n0), int(n1))
                   for eid, n0, n1 in ConnectivityMatrix_line.astype(int)}

for eid in hinge_edges:
    n0, n1 = edgeid_to_nodes[eid]
    key = tuple(sorted((n0, n1)))
    oppA, oppB = edge_to_opps[key]   # the two “third” vertices
    hinge_quads.append([eid, n0, n1, oppA, oppB])

# Convert to a NumPy array
HingeQuads = np.array(hinge_quads, dtype=int)

print("edgeID, node0, node1, node2, node3")
print(HingeQuads)

HingeQuads



# %% Arrange HingeQuads nodes order to get HingeQuads_order

HingeQuads_order = []
for eid, n0, n1, oppA, oppB in HingeQuads:
    # fetch coordinates
    x0 = X0_4columns[n0, 1:4]
    x1 = X0_4columns[n1, 1:4]
    x2 = X0_4columns[oppA,1:4]
    x3 = X0_4columns[oppB,1:4]

    # build edge and triangle vectors
    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0

    # normals for the two triangles sharing the hinge
    n0_v = np.cross(m_e0, m_e1)   # from triangle (n0,n1,oppA)
    n1_v = np.cross(m_e2, m_e0)   # from triangle (oppB,n0,n1)

    # if both normals point upward (z >= 0) keep order,
    # else swap the two opposite‐vertices (oppA ↔ oppB)
    if (n0_v[2] < 0) and (n1_v[2] < 0):
        row = [eid, n0, n1, oppB, oppA]

    if (n0_v[2] > 0) and (n1_v[2] > 0):
        row = [eid, n0, n1, oppA, oppB]
        
    if (n0_v[2])*(n1_v[2])<0:
        raise ValueError("n0_z*n1_z<0, update ordering criterion!")
        
    HingeQuads_order.append(row)

HingeQuads_order = np.array(HingeQuads_order, dtype=int)

print("edgeID, node0, node1, node2, node3 (ordered):")
print(HingeQuads_order)



n_hinges    = HingeQuads_order.shape[0]
# the “central” edge‐IDs for each hinge quad
hinge_eids  = HingeQuads_order[:,0]          # shape (n_hinges,)

# bending stiffness per hinge
#  kb = Y * h^3 / (√3·6)
# kb_array = Y_array[hinge_eids] * (h**3) / (np.sqrt(3) * 6.0)


kb_array = np.where(Y_array[hinge_eids] == Yhard, kb12,
           np.where(Y_array[hinge_eids] == Ysoft, kb1, 0.0))





ks_array = ks_array * FactorKs
kb_array = kb_array * FactorKb



# sanity check
assert kb_array.shape == (n_hinges,)


# %% Def. functions: bending dihedral angle theta, gradient d(theta)/dx, hessian d^2(theta)/dx^2.

def gradTheta(x0, x1 = None, x2 = None, x3 = None):

    if np.size(x0) == 12:  # Allow another type of input where x0 contains all the info
      x1 = x0[3:6]
      x2 = x0[6:9]
      x3 = x0[9:12]
      x0 = x0[0:3]

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0
    m_e3 = x2 - x1
    m_e4 = x3 - x1

    m_cosA1 = np.dot(m_e0, m_e1) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_cosA2 = np.dot(m_e0, m_e2) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_cosA3 = -np.dot(m_e0, m_e3) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_cosA4 = -np.dot(m_e0, m_e4) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_sinA1 = np.linalg.norm(np.cross(m_e0, m_e1)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_sinA2 = np.linalg.norm(np.cross(m_e0, m_e2)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_sinA3 = -np.linalg.norm(np.cross(m_e0, m_e3)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_sinA4 = -np.linalg.norm(np.cross(m_e0, m_e4)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_nn1 = np.cross(m_e0, m_e3)
    m_nn1 = m_nn1 / np.linalg.norm(m_nn1)
    m_nn2 = -np.cross(m_e0, m_e4)
    m_nn2 = m_nn2 / np.linalg.norm(m_nn2)

    m_h1 = np.linalg.norm(m_e0) * m_sinA1
    m_h2 = np.linalg.norm(m_e0) * m_sinA2
    m_h3 = -np.linalg.norm(m_e0) * m_sinA3  # CORRECTION
    m_h4 = -np.linalg.norm(m_e0) * m_sinA4  # CORRECTION
    m_h01 = np.linalg.norm(m_e1) * m_sinA1
    m_h02 = np.linalg.norm(m_e2) * m_sinA2

    # Initialize the gradient
    gradTheta = np.zeros(12)

    gradTheta[0:3] = m_cosA3 * m_nn1 / m_h3 + m_cosA4 * m_nn2 / m_h4
    gradTheta[3:6] = m_cosA1 * m_nn1 / m_h1 + m_cosA2 * m_nn2 / m_h2
    gradTheta[6:9] = -m_nn1 / m_h01
    gradTheta[9:12] = -m_nn2 / m_h02

    return gradTheta

def hessTheta(x0, x1 = None, x2 = None, x3 = None):

    if np.size(x0) == 12:  # Allow another type of input where x0 contains all the info
      x1 = x0[3:6]
      x2 = x0[6:9]
      x3 = x0[9:12]
      x0 = x0[0:3]

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0
    m_e3 = x2 - x1
    m_e4 = x3 - x1

    m_cosA1 = np.dot(m_e0, m_e1) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_cosA2 = np.dot(m_e0, m_e2) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_cosA3 = -np.dot(m_e0, m_e3) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_cosA4 = -np.dot(m_e0, m_e4) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_sinA1 = np.linalg.norm(np.cross(m_e0, m_e1)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_sinA2 = np.linalg.norm(np.cross(m_e0, m_e2)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_sinA3 = -np.linalg.norm(np.cross(m_e0, m_e3)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_sinA4 = -np.linalg.norm(np.cross(m_e0, m_e4)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_nn1 = np.cross(m_e0, m_e3)
    m_nn1 /= np.linalg.norm(m_nn1)
    m_nn2 = -np.cross(m_e0, m_e4)
    m_nn2 /= np.linalg.norm(m_nn2)

    m_h1 = np.linalg.norm(m_e0) * m_sinA1
    m_h2 = np.linalg.norm(m_e0) * m_sinA2
    m_h3 = -np.linalg.norm(m_e0) * m_sinA3
    m_h4 = -np.linalg.norm(m_e0) * m_sinA4
    m_h01 = np.linalg.norm(m_e1) * m_sinA1
    m_h02 = np.linalg.norm(m_e2) * m_sinA2

    # Gradient of Theta (as an intermediate step)
    grad_theta = np.zeros((12, 1))
    grad_theta[0:3] = (m_cosA3 * m_nn1 / m_h3 + m_cosA4 * m_nn2 / m_h4).reshape(-1, 1)
    grad_theta[3:6] = (m_cosA1 * m_nn1 / m_h1 + m_cosA2 * m_nn2 / m_h2).reshape(-1, 1)
    grad_theta[6:9] = (-m_nn1 / m_h01).reshape(-1, 1)
    grad_theta[9:12] = (-m_nn2 / m_h02).reshape(-1, 1)

    # Intermediate matrices for Hessian
    m_m1 = np.cross(m_nn1, m_e1) / np.linalg.norm(m_e1)
    m_m2 = -np.cross(m_nn2, m_e2) / np.linalg.norm(m_e2)
    m_m3 = -np.cross(m_nn1, m_e3) / np.linalg.norm(m_e3)
    m_m4 = np.cross(m_nn2, m_e4) / np.linalg.norm(m_e4)
    m_m01 = -np.cross(m_nn1, m_e0) / np.linalg.norm(m_e0)
    m_m02 = np.cross(m_nn2, m_e0) / np.linalg.norm(m_e0)

    # Hessian matrix components
    M331 = m_cosA3 / (m_h3 ** 2) * np.outer(m_m3, m_nn1)
    M311 = m_cosA3 / (m_h3 * m_h1) * np.outer(m_m1, m_nn1)
    M131 = m_cosA1 / (m_h1 * m_h3) * np.outer(m_m3, m_nn1)
    M3011 = m_cosA3 / (m_h3 * m_h01) * np.outer(m_m01, m_nn1)
    M111 = m_cosA1 / (m_h1 ** 2) * np.outer(m_m1, m_nn1)
    M1011 = m_cosA1 / (m_h1 * m_h01) * np.outer(m_m01, m_nn1)

    M442 = m_cosA4 / (m_h4 ** 2) * np.outer(m_m4, m_nn2)
    M422 = m_cosA4 / (m_h4 * m_h2) * np.outer(m_m2, m_nn2)
    M242 = m_cosA2 / (m_h2 * m_h4) * np.outer(m_m4, m_nn2)
    M4022 = m_cosA4 / (m_h4 * m_h02) * np.outer(m_m02, m_nn2)
    M222 = m_cosA2 / (m_h2 ** 2) * np.outer(m_m2, m_nn2)
    M2022 = m_cosA2 / (m_h2 * m_h02) * np.outer(m_m02, m_nn2)

    B1 = 1 / np.linalg.norm(m_e0) ** 2 * np.outer(m_nn1, m_m01)
    B2 = 1 / np.linalg.norm(m_e0) ** 2 * np.outer(m_nn2, m_m02)

    N13 = 1 / (m_h01 * m_h3) * np.outer(m_nn1, m_m3)
    N24 = 1 / (m_h02 * m_h4) * np.outer(m_nn2, m_m4)
    N11 = 1 / (m_h01 * m_h1) * np.outer(m_nn1, m_m1)
    N22 = 1 / (m_h02 * m_h2) * np.outer(m_nn2, m_m2)
    N101 = 1 / (m_h01 ** 2) * np.outer(m_nn1, m_m01)
    N202 = 1 / (m_h02 ** 2) * np.outer(m_nn2, m_m02)

    # Initialize Hessian of Theta
    hess_theta = np.zeros((12, 12))

    hess_theta[0:3, 0:3] = mmt(M331) - B1 + mmt(M442) - B2
    hess_theta[0:3, 3:6] = M311 + M131.T + B1 + M422 + M242.T + B2
    hess_theta[0:3, 6:9] = M3011 - N13
    hess_theta[0:3, 9:12] = M4022 - N24
    hess_theta[3:6, 3:6] = mmt(M111) - B1 + mmt(M222) - B2
    hess_theta[3:6, 6:9] = M1011 - N11
    hess_theta[3:6, 9:12] = M2022 - N22
    hess_theta[6:9, 6:9] = -mmt(N101)
    hess_theta[9:12, 9:12] = -mmt(N202)

    # Make the Hessian symmetric
    hess_theta[3:6, 0:3] = hess_theta[0:3, 3:6].T
    hess_theta[6:9, 0:3] = hess_theta[0:3, 6:9].T
    hess_theta[9:12, 0:3] = hess_theta[0:3, 9:12].T
    hess_theta[6:9, 3:6] = hess_theta[3:6, 6:9].T
    hess_theta[9:12, 3:6] = hess_theta[3:6, 9:12].T

    return hess_theta


def test_gradTheta():
  # Randomly choose four points
  x0 = np.random.rand(3)
  x1 = np.random.rand(3)
  x2 = np.random.rand(3)
  x3 = np.random.rand(3)

  # Combine the points into a single array
  X_0 = np.concatenate([x0, x1, x2, x3])

  # Analytical gradient of theta
  grad = gradTheta(X_0)

  # Numerical gradient calculation
  gradNumerical = np.zeros(12)
  dx = 1e-6
  theta_0 = getTheta(X_0)

  # Loop through each element to compute the numerical gradient
  for c in range(12):
      X_0dx = X_0.copy()
      X_0dx[c] += dx
      theta_dx = getTheta(X_0dx)
      gradNumerical[c] = (theta_dx - theta_0) / dx

  # Plotting the analytical vs numerical gradients
  plt.figure()
  plt.plot(range(1, len(grad) + 1), grad, 'ro', label='Analytical')
  plt.plot(range(1, len(grad) + 1), gradNumerical, 'b^', label='Numerical')
  plt.xlabel('Index Number')
  plt.ylabel('Gradient of theta, F_{i}')
  plt.legend()
  plt.grid(True)
  plt.show()


# test_gradTheta()


def test_hessTheta():
    # Randomly choose four points
    x0 = np.random.rand(3)
    x1 = np.random.rand(3)
    x2 = np.random.rand(3)
    x3 = np.random.rand(3)

    # Assemble the four vectors into a long vector
    X_0 = np.concatenate([x0, x1, x2, x3])

    # Analytical gradient and Hessian of theta
    grad_theta_0 = gradTheta(X_0)  # Replace with your gradTheta function
    hess = hessTheta(X_0)          # Replace with your hessTheta function

    # Numerical Hessian calculation
    hess_numerical = np.zeros((12, 12))
    dx = 1e-6

    for c in range(12):
        X_0dx = X_0.copy()
        X_0dx[c] += dx
        grad_theta_dx = gradTheta(X_0dx)
        dHess = (grad_theta_dx - grad_theta_0) / dx
        hess_numerical[c, :] = dHess

    # Plot the results
    plt.figure()
    plt.plot(np.arange(len(hess.flatten())), hess.flatten(), 'ro', label='Analytical')
    plt.plot(np.arange(len(hess_numerical.flatten())), hess_numerical.flatten(), 'b^', label='Numerical')
    plt.xlabel('Index Number')
    plt.ylabel('Hessian of theta, J_{ij}')
    plt.legend()
    plt.grid(True)
    plt.show()


# test_hessTheta()


# %% Def. functions: gradient, Hessian of bending energy: dEb/dx, d^2(Eb)/dx^2. 

def getEb_Shell(x0, x1=None, x2=None, x3=None, theta_bar=0, kb=1):
    """
    Compute the bending energy for a shell.

    Returns:
    E (scalar): Bending energy.
    """
    # Allow another type of input where x0 contains all the information
    if np.size(x0) == 12:
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[:3]

    # Compute theta, gradient, and Hessian
    theta = getTheta(x0, x1, x2, x3)  # Replace with your getTheta function in Python
    grad = gradTheta(x0, x1, x2, x3)  # Replace with your gradTheta function in Python

    # E = 0.5 * kb * (theta-thetaBar)^2
    E = 0.5 * kb * (theta - theta_bar) ** 2

    return E

def gradEb_hessEb_Shell(x0, x1=None, x2=None, x3=None, theta_bar=0, kb=1):
    """
    Compute the gradient and Hessian of the bending energy for a shell.

    Parameters:
    x0 (array): Can either be a 3-element array (single point) or a 12-element array.
    x1, x2, x3 (arrays): Optional, 3-element arrays specifying points.
    theta_bar (float): Reference angle.
    kb (float): Bending stiffness.

    Returns:
    dF (array): Gradient of the bending energy.
    dJ (array): Hessian of the bending energy.
    """
    # Allow another type of input where x0 contains all the information
    if np.size(x0) == 12:
        x1 = x0[3:6]
        x2 = x0[6:9]
        x3 = x0[9:12]
        x0 = x0[:3]

    # Compute theta, gradient, and Hessian
    theta = getTheta(x0, x1, x2, x3)  # Replace with your getTheta function in Python
    grad = gradTheta(x0, x1, x2, x3)  # Replace with your gradTheta function in Python

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = dE/dx = 2 * (theta-thetaBar) * gradTheta
    dF = 0.5 * kb * (2 * (theta - theta_bar) * grad)

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = 0.5 * kb * (2 (theta-thetaBar) d theta/dx)
    # J = dF/dx = 0.5 * kb * [ 2 (d theta / dx) transpose(d theta/dx) +
    #       2 (theta-thetaBar) (d^2 theta/ dx^2 ) ]
    hess = hessTheta(x0, x1, x2, x3)  # Replace with your hessTheta function in Python
    dJ = 0.5 * kb * (2 * np.outer(grad, grad) + 2 * (theta - theta_bar) * hess)

    return dF, dJ



def test_gradEb():
  # Randomly choose four points
  x0 = np.random.rand(3)
  x1 = np.random.rand(3)
  x2 = np.random.rand(3)
  x3 = np.random.rand(3)

  theta_bar = np.random.rand() # Random undeformed angle
  kb = np.random.rand() # Random bending stiffness

  # Combine the points into a single array
  X_0 = np.concatenate([x0, x1, x2, x3])

  # Analytical gradient of theta
  grad, _ = gradEb_hessEb_Shell(x0=X_0, theta_bar=theta_bar,kb=kb)

  # Numerical gradient calculation
  gradNumerical = np.zeros(12)
  dx = 1e-6
  E_0 = getEb_Shell(X_0, theta_bar=theta_bar,kb=kb)

  # Loop through each element to compute the numerical gradient
  for c in range(12):
      X_0dx = X_0.copy()
      X_0dx[c] += dx
      E_dx = getEb_Shell(x0=X_0dx, theta_bar=theta_bar,kb=kb)
      gradNumerical[c] = (E_dx - E_0) / dx

  # Plotting the analytical vs numerical gradients
  plt.figure()
  plt.plot(range(1, len(grad) + 1), grad, 'ro', label='Analytical')
  plt.plot(range(1, len(grad) + 1), gradNumerical, 'b^', label='Numerical')
  plt.xlabel('Index Number')
  plt.ylabel('Gradient of theta, F_{i}')
  plt.legend()
  plt.grid(True)
  plt.show()


# test_gradEb()


def test_hessEb():
    # Randomly choose four points
    x0 = np.random.rand(3)
    x1 = np.random.rand(3)
    x2 = np.random.rand(3)
    x3 = np.random.rand(3)

    theta_bar = np.random.rand() # Random undeformed angle
    kb = np.random.rand() # Random bending stiffness

    # Assemble the four vectors into a long vector
    X_0 = np.concatenate([x0, x1, x2, x3])

    # Analytical gradient and Hessian of theta
    grad_Eb_0, hess_Eb = gradEb_hessEb_Shell(x0=X_0, theta_bar=theta_bar,kb=kb)

    # Numerical Hessian calculation
    hess_numerical = np.zeros((12, 12))
    dx = 1e-6

    for c in range(12):
        X_0dx = X_0.copy()
        X_0dx[c] += dx
        grad_Eb_dx, _ = gradEb_hessEb_Shell(x0=X_0dx, theta_bar=theta_bar,kb=kb)
        dHess = (grad_Eb_dx - grad_Eb_0) / dx
        hess_numerical[c, :] = dHess

    # Plot the results
    plt.figure()
    plt.plot(np.arange(len(hess_Eb.flatten())), hess_Eb.flatten(), 'ro', label='Analytical')
    plt.plot(np.arange(len(hess_numerical.flatten())), hess_numerical.flatten(), 'b^', label='Numerical')
    plt.xlabel('Index Number')
    plt.ylabel('Hessian of theta, J_{ij}')
    plt.legend()
    plt.grid(True)
    plt.show()


# test_hessEb()




# %% Def. functions: gradient, Hessian of COUPLED bending energy: dEb_coupled/dx, d^2(Eb_coupled)/dx^2. 



def fun_DEps_grad_hess(xloc: np.ndarray,
                       nodes: List[int],
                       edge_length_fn: Callable[[int,int], float]
                       ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    For one hinge‐quad with global nodes [n0,n1,oppA,oppB]:
      Δε = +ε(n0–oppA)
           −ε(n0–oppB)
           +ε(n1–oppA)
           −ε(n1–oppB)
    Returns (Deps, GDeps (12,), HDeps (12×12)) matching that exact sequence.
    """
    # unpack your four node‐IDs
    n0, n1, oppA, oppB = nodes

    # build pairs in the exact order you want:
    pairs = [
      (n0, oppA),
      (n0, oppB),
      (n1, oppA),
      (n1, oppB),
    ]
    signs = [+1,  -1,  +1,  -1]

    Deps  = 0.0
    GDeps = np.zeros(12)
    HDeps = np.zeros((12,12))

    for (a,b), s in zip(pairs, signs):
        # find their local slots in xloc
        ia = nodes.index(a)
        ib = nodes.index(b)
        loc0 = slice(3*ia,   3*ia+3)
        loc1 = slice(3*ib,   3*ib+3)

        x0 = xloc[loc0]
        x1 = xloc[loc1]
        L0 = edge_length_fn(a, b)

        # 1) scalar stretch and accumulate
        eps_ab = get_strain_stretch_edge2D3D(x0, x1, L0)
        Deps  += s * eps_ab

        # 2) its grad & hess
        dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)

        # 3) scatter into our 12-vector
        GDeps[loc0] +=  s * dG_e[0:3]
        GDeps[loc1] +=  s * dG_e[3:6]

        idx = list(range(3*ia,3*ia+3)) + list(range(3*ib,3*ib+3))
        HDeps[np.ix_(idx,idx)] +=  s * dH_e

    return Deps, GDeps, HDeps



def fun_DEps_grad_hess_thermal(xloc: np.ndarray,
                               nodes: List[int],
                               edge_length_fn: Callable[[int,int], float],
                               eps_th_vector: np.ndarray,
                               edge_dict: Dict[Tuple[int,int],int]
                               ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    For one hinge‐quad with global nodes [n0,n1,oppA,oppB]:
      Δε = +ε(n0–oppA)
           −ε(n0–oppB)
           +ε(n1–oppA)
           −ε(n1–oppB)
    Like fun_DEps_grad_hess, but for Δε_mech - Δε_th:
        Δε_total = Σ_signs [ ε_mech(a–b) - ε_th[eid(a,b)] ]
    Returns (Deps_total, GDeps, HDeps).
    """
    # unpack your four node‐IDs
    n0, n1, oppA, oppB = nodes

    # build pairs in the exact order you want:
    pairs = [
      (n0, oppA),
      (n0, oppB),
      (n1, oppA),
      (n1, oppB),
    ]
    signs = [+1,  -1,  +1,  -1]

    Deps  = 0.0
    GDeps = np.zeros(12)
    HDeps = np.zeros((12,12))

    for (a,b), s in zip(pairs, signs):
        # find their local slots in xloc
        ia = nodes.index(a)
        ib = nodes.index(b)
        loc0 = slice(3*ia,   3*ia+3)
        loc1 = slice(3*ib,   3*ib+3)

        x0 = xloc[loc0]
        x1 = xloc[loc1]
        L0 = edge_length_fn(a, b)

        # mechanical stretch
        eps_mech = get_strain_stretch_edge2D3D(x0, x1, L0)
        # look up the thermal strain for that same edge
        eid      = edge_dict[tuple(sorted((a,b)))]
        eps_th   = eps_th_vector[eid]

        # accumulate signed (mech - th)
        Deps += s * (eps_mech - eps_th)

        # 2) its grad & hess
        dG_e, dH_e = grad_and_hess_strain_stretch_edge3D(x0, x1, L0)

        # 3) scatter into our 12-vector
        GDeps[loc0] +=  s * dG_e[0:3]
        GDeps[loc1] +=  s * dG_e[3:6]

        idx = list(range(3*ia,3*ia+3)) + list(range(3*ib,3*ib+3))
        HDeps[np.ix_(idx,idx)] +=  s * dH_e

    return Deps, GDeps, HDeps


def fun_coupled_Ebend_grad_hess(
        xloc: np.ndarray,
        nodes: List[int],
        edge_length_fn: Callable[[int,int], float],
        beta: float,
        kb: float
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    For one hinge‑quad with 4 local nodes (in the order [n0,n1,oppA,oppB]) and 
    xloc.shape==(12,), compute the gradient & Hessian of
      ½·kb·[θ − β·Δε]²
    where Δε = +ε(n0–oppA)
              −ε(n0–oppB)
              +ε(n1–oppA)
              −ε(n1–oppB).
    Returns:
      dG (12,) : ∂E/∂xloc
      dH (12,12) : ∂²E/∂xloc²
    """
    # 1) pure‐bending pieces
    θ   = getTheta(xloc)
    gθ  = gradTheta(xloc)
    Hθ  = hessTheta(xloc)

    # 2) the Δε, its grad & hess in the exact ordering you want
    Deps, GDeps, HDeps = fun_DEps_grad_hess(xloc, nodes, edge_length_fn)

    # 3) form force‐like and stiffness‐like pieces
    f_h = θ - beta * Deps           # scalar
    C_h = gθ  - beta * GDeps        # shape (12,)

    dG = kb * f_h * C_h
    dH = kb * ( np.outer(C_h, C_h)
              + f_h*(Hθ - beta*HDeps) )

    return dG, dH


def fun_coupled_Ebend_grad_hess_thermal(
        xloc: np.ndarray,
        nodes: List[int],
        edge_length_fn: Callable[[int,int], float],
        beta: float,
        kb: float,
        eps_th: np.ndarray,
        edge_dict: Dict[Tuple[int,int],int]
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    For one hinge‑quad with 4 local nodes (in the order [n0,n1,oppA,oppB]) and 
    xloc.shape==(12,), compute the gradient & Hessian of
      ½·kb·[θ − β·Δε]²
    where Δε = +ε(n0–oppA)
              −ε(n0–oppB)
              +ε(n1–oppA)
              −ε(n1–oppB).
    Change kb according to dihedral angle sign.
    Returns:
      dG (12,) : ∂E/∂xloc
      dH (12,12) : ∂²E/∂xloc²
    """
    # 1) pure‐bending pieces
    θ   = getTheta(xloc)
    gθ  = gradTheta(xloc)
    Hθ  = hessTheta(xloc)

    #  # Not converge for 1.1 when ks_hard=100*ks_soft
    #  # converge for 1.05
    Factor_kb = 1.0
    kb_angle  = Factor_kb*kb if θ > 0 else kb
    
    # 2) the Δε, its grad & hess in the exact ordering you want
    Deps, GDeps, HDeps = fun_DEps_grad_hess_thermal(xloc, nodes, edge_length_fn, eps_th, edge_dict)

    # 3) form force‐like and stiffness‐like pieces
    f_h = θ - beta * Deps           # scalar
    C_h = gθ  - beta * GDeps        # shape (12,)

    dG = kb_angle * f_h * C_h
    dH = kb_angle * ( np.outer(C_h, C_h)
                    + f_h*(Hθ - beta*HDeps) )

    return dG, dH


# %% Def. a class: calculate, assembly of elemental gradient, Hessian to global grad, Hess.

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
    
    
    
def test_ElasticGHEdges(energy_choice,theta_bar,Nedges,NP_total,Ndofs,connectivity,l0_ref, 
                        X0, iPrint, HingeQuads_order ):
    
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
        connectivity  = ConnectivityMatrix_line,
        l0_ref        = L0,
        ks            = EA,
        hinge_quads   = HingeQuads_order,
        theta_bar     = theta_bar,
        kb            = 1.0,
        h             = h,
        model_choice  = 1
    )
    
    G, H = struct1.computeGradientHessian(q)

    # (6) Print results
    if iPrint:
        np.set_printoptions(precision=4, suppress=True)
        print("Assembled force vector G:")
        print(G)
        print("\nAssembled stiffness matrix H:")
        print(H)
    
    # Need to verify the assembly. Done
    # Then modify FDM_ElasticForce to verify. Done


if iTest ==1:
    test_ElasticGHEdges(
        energy_choice    = 3,
        theta_bar        = 0.0,
        Nedges           = Nedges,
        NP_total         = NP_total,
        Ndofs            = Ndofs,
        connectivity     = ConnectivityMatrix_line,
        l0_ref           = L0,
        X0               = X0,
        iPrint           = iPrint,
        HingeQuads_order = HingeQuads_order)
    
    
def FDM_ElasticGHEdges(energy_choice,Nedges,NP_total,Ndofs,connectivity,l0_ref, 
                       X0, iPrint, HingeQuads_order, theta_bar):
    
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


# Verify the Grad and Hess from class ElasticGHEdges.
if iTest ==1:
    FDM_ElasticGHEdges(
        energy_choice    = 3,
        Nedges           = Nedges,
        NP_total         = NP_total,
        Ndofs            = Ndofs,
        connectivity     = ConnectivityMatrix_line,
        l0_ref           = L0,
        X0               = X0,
        iPrint           = iPrint,
        HingeQuads_order = HingeQuads_order,
        theta_bar        = 0.0)



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
    eps:float=1e-6
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


if iTest ==1:
    FDM_ElasticGHEdgesCoupled(
        energy_choice=4,                # coupled bending
        Nedges=Nedges,
        NP_total=NP_total,
        Ndofs=Ndofs,
        connectivity=ConnectivityMatrix_line,
        l0_ref=L0,
        hinge_quads=HingeQuads_order,
        theta_bar=0.0,
        ks=1.0,
        kb=1.0,
        beta=1.0,
        h=0.1,
        model_choice=1,
        X0=X0,
        iPrint=True,
        eps=1e-6)



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


# For the beam mesh, Hessian error:
# Relative Frobenius‐norm error: 6.579137264106082e-05
# L_inf max abs error: 3.1800e-01,
# Root-mean-square (RMS) error: 1.9231e-03



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
    eps:float=1e-6
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


if iTest ==1:
    FDM_ElasticGHEdgesCoupledThermal(
        energy_choice=4,                # coupled bending
        Nedges=Nedges,
        NP_total=NP_total,
        Ndofs=Ndofs,
        connectivity=ConnectivityMatrix_line,
        l0_ref=L0,
        hinge_quads=HingeQuads_order,
        theta_bar=0.0,
        ks=1.0,
        kb=1.0,
        beta=1.0,
        h=0.1,
        eps_th_vector=eps_th_vector,
        model_choice=1,
        X0=X0,
        iPrint=True,
        eps=1e-6)


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





# %% Main program

# Add options to have bending energy. 05-01-25, 6:26pm

# Geometry info from mesh
X0_4columns
ConnectivityMatrix_line
Triangles

NP_total
Nedges
Ntriangles
Ndofs

X0
L0


bc1 = BoundaryConditions3D(Ndofs)


theta_bar=0.0
model_choice = 1
beta=1.0

# elastic_model1 = ElasticGHEdges(
#     energy_choice = 3,
#     Nedges        = Nedges,
#     NP_total      = NP_total,
#     Ndofs         = Ndofs,
#     connectivity  = ConnectivityMatrix_line,
#     l0_ref        = L0,
#     ks            = ks,
#     hinge_quads   = HingeQuads_order,
#     theta_bar     = theta_bar,
#     kb            = kb,
#     h             = h,
#     model_choice  = 1
# )


# elastic_model0 = ElasticGHEdgesCoupled(
# 	energy_choice = 4,
#     Nedges        = Nedges,
#     NP_total      = NP_total,
#     Ndofs         = Ndofs,
#     connectivity  = ConnectivityMatrix_line,
#     l0_ref        = L0,
#     ks_array      = ks_array,
#     hinge_quads   = HingeQuads_order,
#     theta_bar     = theta_bar,
#     kb_array      = kb_array,
#     h             = h,
#     beta          = beta,
#     model_choice  = 1
#     )      


# Verify Grad and Hess of whole system coupled energy.
# At equilibrium, undeformed q0 -> zero strains and dihedral angles -> ∇E=0.
# So add perturbations to have nonzero ∇E values. 05-07-25
# if iTest ==1:   
#     q0 = X0.copy()
#     q1 = q0 + 1e-6*np.random.randn(*q0.shape)
#     test_fun_total_energy_grad_hess(elastic_model0,q1,iPlot) # coupled energy



elastic_model1 = ElasticGHEdgesCoupledThermal(
    energy_choice = 4,
    Nedges        = Nedges,
    NP_total      = NP_total,
    Ndofs         = Ndofs,
    connectivity  = ConnectivityMatrix_line,
    l0_ref        = L0,
    ks_array      = ks_array,
    hinge_quads   = HingeQuads_order,
    theta_bar     = theta_bar,
    kb_array      = kb_array,
    beta          = beta,
    epsilon_th    = eps_th_vector,
    model_choice  = 1
)

if iTest ==1:
    q0 = X0.copy()
    q1 = X0 + 1e-6*np.random.randn(*q0.shape)
    test_fun_total_energy_grad_hess_thermal(elastic_model1, q1, iplot=1)


# 3D time‐stepper



# totalMass   = 0.001164      # total mass of the structure
# massVector  = np.full(Ndofs, totalMass/NP_total)  # lumped equally per node


if iMesh ==1:
    NodalMass=1e-7

if iMesh ==2:
    NodalMass=1e-7 #-6 for case 3 # 0.000001 works when kb=1/10 ref value.

if iMesh ==3:
    NodalMass=1e-7
    
massVector  = np.full(Ndofs, NodalMass)  # lumped equally per node




qtol        = 1e-5     
maxIter     = 20     

if iGravity == 0:
    g_vec = np.array([0.0, 0.0, 0.0]) 
    
if iGravity == 1:
    g_vec = np.array([0.0, 0.0, -9.81])  # gravity in -z


g_vec=-g_vec   # The z axis is inverse of physical world. 05-21-25

totalTime    = 1.0
dt_min       = 1e-8
dt_max       = 0.1

dt           = 0.05 #0.05 #.02 For eps_th=-0.5

n_record     = 2

n_hinges       = HingeQuads_order.shape[0]
Q_history      = np.zeros((n_record, Ndofs))
R_history      = np.zeros((n_record, Ndofs))
strain_history = np.zeros((n_record, Nedges))
stress_history = np.zeros((n_record, Nedges))
theta_history  = np.zeros((n_record, n_hinges))
length_history = np.zeros((n_record, Nedges))


# stepper3D = timeStepper3D(
#     massVector, dt, qtol, maxIter,
#     g_vec, bc1, elastic_model1, X0)

stepper3D = timeStepper3D_static(
    massVector, dt, qtol, maxIter,
    g_vec, bc1, elastic_model1, X0)

# stepper3D = timeStepper3D_static_gravity(
#     massVector, dt, qtol, maxIter,
#     g_vec, bc1, elastic_model1, X0)


q_old = X0.copy()               # start at undeformed reference
u_old = np.zeros(Ndofs)         # zero initial velocity
a_old = np.zeros(Ndofs)         # zero initial acceleration


# store initial state at step=0
record_step(
    step=0,
    q_new=q_old,
    elastic_model=elastic_model1,
    connectivity=ConnectivityMatrix_line,
    L0=L0,
    ks_array=ks_array,
    hinge_quads=HingeQuads_order,
    Q_history=Q_history,
    R_history=R_history,
    length_history=length_history, 
    strain_history=strain_history,
    stress_history=stress_history,
    theta_history=theta_history)


print("---Netwon iteration starts---")

step_log = []
time_log = []
t    = 0.0
step = 0
while True:
# for step in range(1, n_steps+1):
# for t in np.arange(0, totalTime, dt):
    
    if t + dt > totalTime:
        dt = totalTime - t
    stepper3D.dt = dt
    
    t_next = t + dt
    elastic_model1.eps_th = eps_th_vector * (t_next/totalTime)
    print(f"t = {t_next:.8f}, g={stepper3D.g}\n") #" eps_th = {elastic_model1.eps_th}")

    # if iMesh == 0:
    #     fun_BC_evolution_example_5nodes(bc1, X0, NP_total, t_next, 1e-8)

    # if iMesh in (1, 2):
    #     fixedNodes = fun_BC_3D_hold_center(bc1, X0, NP_total, half_x=0.15, half_y=0.15)

    # if iMesh == 3:
    #     fun_BC_beam_PointLoad(bc1, X0, NP_total, t=t_next, x_thresh=0.21, disp_right=-0.001, tol=1e-8)
    
    # if iMesh == 4:
    #     fun_BC_4nodes(bc1)
    if iMesh == 1:
        fixedNodes = fun_BC_3D_hold_center(bc1, X0, NP_total, half_x=0.01, half_y=0.01)

    if iMesh == 2:
        fixedNodes = fun_BC_peanut(bc1, X0, NP_total,
                                   x_min = 0.0577, x_max = 0.0987,
                                   y_min = 0.0458, y_max = 0.0538,
                                   node_region_fn = region_fn)  
        
        # if n_spokes == 6:
        #     fixedNodes = fun_BC_peanut(bc1, X0, NP_total,
        #                                x_min = 0.145, x_max = 0.248,
        #                                y_min = 0.115, y_max = 0.135,
        #                                node_region_fn = region_fn)  

    if iMesh == 3:
        temp = 0.01
        fixedNodes = fun_BC_peanut(bc1, X0, NP_total,
                             x_min = Pattern_center[0]-temp, x_max = Pattern_center[0]+temp,
                             y_min = Pattern_center[1]-temp, y_max = Pattern_center[1]+temp,
                             node_region_fn = region_fn)
        
        
        
    if iMesh in (5, 6, 7, 8):
        fixedNodes = fun_BC_3D_hold_center(bc1, X0, NP_total, half_x=0.01, half_y=0.01)

            
    # if iMesh == 9:
    #     ## rectangle with a stripe 
    #     fixedNodes = fun_BC_peanut(bc1, X0, NP_total,
    #                          x_min = 0.05, x_max = 0.25,
    #                          y_min = 0.0, y_max = ys.max(),
    #                          node_region_fn = stripe_region)
        
    if iPrint:
        fixedVals, fixedIdxs, freeIdxs = bc1.getBoundaryConditions()
        print("Fixed DOF indices: ", fixedIdxs)
        print("Fixed DOF values: ", *("0.0" if v == 0.0 else f"{v:.4e}" for v in fixedVals))
        # print("Free DOF indices: ", freeIdxs)
        print("Fixed nodes: ", fixedNodes)
        

    q_new, converged = stepper3D.simulate(q_old, q_old, u_old, a_old)
    # q_new, u_new, a_new, converged = stepper3D.simulate(
    #     q_old, q_old, u_old, a_old)
    
    if not converged:
        if dt > dt_min:
            dt = max(dt * 0.5, dt_min)
            print(f"Newton failed at t={t_next:.8f}, reducing dt to {dt:.8f} and retrying")
            continue
        else:                
            print(f"Not converged at t = {t:.8f} s, saving failure state")
            record_step(
                1,
                q_new,
                elastic_model1,
                ConnectivityMatrix_line,
                L0,
                ks_array,
                HingeQuads_order,
                Q_history,
                R_history,
                length_history,
                strain_history,
                stress_history,
                theta_history)  
            break
        
    else:
        t     = t_next
        step += 1
        # q_old, u_old, a_old = q_new.copy(), u_new.copy(), a_new.copy()
        q_old = q_new.copy()
        
        step_log.append(step)
        time_log.append(t)
        
        if stepper3D.last_num_iters < 5:
            dt = min(dt * 1.05, dt_max)
            print("increase dt\n")
    
    
        if abs(t - totalTime) < 1e-12:
            print("Simulation converged, saving final state")
            record_step(
                1,
                q_new,
                elastic_model1,
                ConnectivityMatrix_line,
                L0,
                ks_array,
                HingeQuads_order,
                Q_history,
                R_history,
                length_history,
                strain_history,
                stress_history,
                theta_history) 
            break
    

print("--- Thermal actuation completed ---")


# plot_truss_2d(
#     q_old,
#     ConnectivityMatrix_line,
#     NP_total=NP_total,
#     title=f"t = {t:.8f} s",
#     show_labels=False)
plot_truss_3d(
    q_old,
    ConnectivityMatrix_line,
    NP_total=NP_total,
    title=f"t = {t:.8f} s",
    show_labels=False)


#  --- Save data to a file ---
mdict = {
    'X0_4columns':          X0_4columns,
    'ConnectivityMatrix_line': ConnectivityMatrix_line,
    'Triangles':            Triangles,
    'NP_total':             NP_total,
    'Nedges':               Nedges,
    'Ntriangles':           Ntriangles,
    'Ndofs':                Ndofs,
    'X0':                   X0,
    'L0':                   L0,
    'ks_array':             ks_array,
    'kb_array':             kb_array,
    'Q_history':            Q_history,
    'R_history':            R_history,
    'strain_history':       strain_history,
    'stress_history':       stress_history,
    'HingeQuads_order':     HingeQuads_order,
    'theta_history':        theta_history,
    'time_log':             time_log,
    'step_log':             step_log,
    'q_old':                q_old,
    'fixedNodes':           fixedNodes
}

# save data with compression
sio.savemat('output_deps_thermal_WithGravity.mat', mdict, do_compression=True)
print("Data saved at ", ", ".join(mdict.keys()))




## ramp-down of external force  

relax_total = 1.0       # seconds over which to turn gravity off
relax_steps = 10
dt_relax     = relax_total / relax_steps
g0           = g_vec  # this was your "physical" gravity

# allocate extra storage if you want to keep logging
extra_records = relax_steps

Q_history      = np.vstack([Q_history,
                            np.zeros((1, Q_history.shape[1]))])
R_history      = np.vstack([R_history,
                            np.zeros((1, R_history.shape[1]))])
strain_history = np.vstack([strain_history,
                            np.zeros((1, strain_history.shape[1]))])
stress_history = np.vstack([stress_history,
                            np.zeros((1, stress_history.shape[1]))])
theta_history  = np.vstack([theta_history,
                            np.zeros((1, theta_history.shape[1]))])
length_history = np.vstack([length_history,
                            np.zeros((1, length_history.shape[1]))])


print("\n=== Gravity ramp‐down ===")
for k in range(1, relax_steps+1):
    # 1) compute new time and gravity
    t_rel = k * dt_relax
    factor = max(0.0, 1.0 - t_rel/relax_total)
    stepper3D.g = g0 * factor
    stepper3D.makeWeight()
    print(f"k = {k:d}, factor={factor:.3f}, g={stepper3D.g} \n")



    # 2) step forward
    q_new, converged = stepper3D.simulate(q_old, q_old, u_old, a_old)
    if not converged:
        raise RuntimeError(f"Failed to converge during gravity‐ramp at step {k}")


    # 4) advance state
    q_old = q_new.copy()
    step_log.append(step+k)
    time_log.append(t+t_rel)
    
    # 3) record
    if k == relax_steps:
        record_step(
                2,
                q_new,
                elastic_model1,
                ConnectivityMatrix_line,
                L0,
                ks_array,
                HingeQuads_order,
                Q_history,
                R_history,
                length_history,
                strain_history,
                stress_history,
                theta_history) 
        

print("Gravity ramp‐down completed")

plot_truss_3d(
    q_old,
    ConnectivityMatrix_line,
    NP_total=NP_total,
    title=f"t = {t:.8f} s",
    show_labels=False)


#  --- Save data to a file ---
mdict = {
    'X0_4columns':          X0_4columns,
    'ConnectivityMatrix_line': ConnectivityMatrix_line,
    'Triangles':            Triangles,
    'NP_total':             NP_total,
    'Nedges':               Nedges,
    'Ntriangles':           Ntriangles,
    'Ndofs':                Ndofs,
    'X0':                   X0,
    'L0':                   L0,
    'ks_array':             ks_array,
    'kb_array':             kb_array,
    'Q_history':            Q_history,
    'R_history':            R_history,
    'strain_history':       strain_history,
    'stress_history':       stress_history,
    'HingeQuads_order':     HingeQuads_order,
    'theta_history':        theta_history,
    'time_log':             time_log,
    'step_log':             step_log,
    'q_old':                q_old,
    'fixedNodes':           fixedNodes,
    'extra_records':        extra_records
}

# save data with compression
sio.savemat('output_deps_thermal_NoG.mat', mdict, do_compression=True)
print("Data saved at ", ", ".join(mdict.keys()))


end = time.perf_counter()
print(f"Elapsed time: {end - start:.4f} s")


