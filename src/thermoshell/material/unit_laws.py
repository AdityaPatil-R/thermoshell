import numpy as np

def get_strain_stretch_edge2D3D(node0, 
                                node1, 
                                l_k):
    # Works for both 2D and 3D.
    # l_k (float): Reference (undeformed) length of the edge.
    edge = node1 - node0
    edgeLen = np.linalg.norm(edge)
    epsX = edgeLen / l_k - 1

    return epsX

def grad_and_hess_strain_stretch_edge3D(node0, 
                                        node1, 
                                        l_k):
    '''
    Compute the gradient and Hessian of the axial stretch of a 3D edge with
    respect to the DOF vector (6 DOFs: x,y,z coords of the two nodes).

    Inputs:
      node0: length-3 array - Position of the first node [x0,y0,z0]
      node1: length-3 array - Position of the second node [x1,y1,z1]
      l_k:   float          - Reference (undeformed) length of the edge

    Outputs:
      dF: length-6 array - Gradient of stretch w.r.t. [x0,y0,z0,x1,y1,z1]
      dJ: 6x6 array      - Hessian of stretch
    '''

    # Edge vector and its length
    edge    = node1 - node0
    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen

    # Axial stretch
    epsX = get_strain_stretch_edge2D3D(node0, node1, l_k)

    # Gradient of stretch w.r.t. the edge-vector
    dF_unit = tangent / l_k
    dF = np.zeros(6)
    dF[0:3] = -dF_unit
    dF[3:6] =  dF_unit

    # Hessian of squared-stretch w.r.t. the edge-vector (3×3)
    I3 = np.eye(3)
    M  = 2.0 / l_k * ((1.0 / l_k - 1.0 / edgeLen) * I3 + (1.0 / edgeLen) * np.outer(edge, edge) / edgeLen ** 2)

    # Convert to Hessian of stretch itself
    if epsX == 0.0:
        M2 = np.zeros_like(M)
    else:
        M2 = 1.0 / (2.0 * epsX) * (M - 2.0 * np.outer(dF_unit, dF_unit))

    # Assemble 6×6 Hessian
    dJ = np.zeros((6,6))
    dJ[0:3, 0:3] =  M2
    dJ[3:6, 3:6] =  M2
    dJ[0:3, 3:6] = -M2
    dJ[3:6, 0:3] = -M2

    return dF, dJ

def grad_and_hess_strain_stretch_edge3D_ZeroStrainStiff(node0, 
                                                        node1, 
                                                        l_k, 
                                                        tol=1e-10):
    '''
    Compute the gradient and Hessian of the axial stretch of a 3D edge with
    respect to the DOF vector (6 DOFs: x,y,z coords of the two nodes).

    Inputs:
      node0: length-3 array - Position of the first node [x0,y0,z0]
      node1: length-3 array - Position of the second node [x1,y1,z1]
      l_k:   float          - Reference (undeformed) length of the edge

    Outputs:
      dF: length-6 array - Gradient of stretch w.r.t. [x0,y0,z0,x1,y1,z1]
      dJ: 6x6 array      - Hessian of stretch
    '''

    # Edge vector and its length
    edge    = node1 - node0
    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen

    # Axial stretch
    epsX = get_strain_stretch_edge2D3D(node0, node1, l_k)

    # Gradient of stretch w.r.t. the edge-vector
    dF_unit = tangent / l_k
    dF = np.zeros(6)
    dF[0:3] = -dF_unit
    dF[3:6] =  dF_unit

    # Hessian of squared-stretch w.r.t. the edge-vector (3x3)
    I3 = np.eye(3)
    M  = 2.0 / l_k * ((1.0 / l_k - 1.0 / edgeLen) * I3 + (1.0 / edgeLen) * np.outer(edge, edge) / edgeLen ** 2)

    # Convert to Hessian of stretch itself
    if abs(epsX) < tol:
        # Small‐strain limit: (I - t t^T)/(L0 * L)
        M2 = (I3 - np.outer(tangent, tangent)) / (l_k * edgeLen)
    else:
        # Full nonlinear Hessian of ε
        M2 = 1.0 / (2.0 * epsX) * (M - 2.0 * np.outer(dF_unit, dF_unit))
        
    # Assemble 6x6 Hessian
    dJ = np.zeros((6,6))
    dJ[0:3, 0:3] =  M2
    dJ[3:6, 3:6] =  M2
    dJ[0:3, 3:6] = -M2
    dJ[3:6, 0:3] = -M2

    return dF, dJ

def fun_grad_hess_energy_stretch_linear_elastic_edge(node0, 
                                                     node1, 
                                                     l_0 = None, 
                                                     ks = None):
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

    # Verify stiffness with norm in small strain
    # With finite strain, geometric stiffness matters
    print("Spring stiffness verify, node1 =", node1)
    print("Sum of squares for 3 DOFs from H:", sum_of_squares)
    print("squared ks/l0:", (ks / l_0) ** 2)

    return G, H


def fun_grad_hess_energy_stretch_linear_elastic_edge_thermal(node0,
                                                             node1, 
                                                             l_0 = None, 
                                                             ks = None, 
                                                             eps_th=0.0):
    # H has two terms, material elastic stiffness + geometric stiffness
    strain_stretch = get_strain_stretch_edge2D3D(node0, node1, l_0) - eps_th
    G_strain, H_strain = grad_and_hess_strain_stretch_edge3D(node0, node1, l_0)

    gradE_strain = ks * strain_stretch * l_0
    hessE_strain = ks * l_0

    G = gradE_strain * G_strain
    H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)
    
    sub_block = H[0:3, 0:3]
    squared = sub_block ** 2
    sum_of_squares = np.sum(squared)

    # Verify stiffness with norm in small strain
    # With finite strain, geometric stiffness matters
    # print("Spring stiffness verify, node1 =", node1)
    # print("Sum of squares for 3DOFs from H:", sum_of_squares)
    # print("Squared ks/l0:", (ks / l_0) ** 2)

    return G, H