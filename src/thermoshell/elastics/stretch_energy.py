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