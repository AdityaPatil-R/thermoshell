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