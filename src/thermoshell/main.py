import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import argparse
from typing import Dict, Tuple, List, Callable
from functools import partial
from itertools import combinations
from collections import defaultdict, Counter

from geometry.mesh_io import load_mesh
from geometry.mesh_props import fun_edge_lengths
from viz.figure_setup import new_fig
from viz.mesh_plots import plot_truss_3d
from viz.thermal_plots import (
    plot_thermal_strain_edges,
    plot_thermal_strain_edges_CustomRange
)
from analysis.material.bilayer import bilayer_flexural_rigidity
from analysis.material.assignment import (
    assign_thermal_strains_contour,
    assign_youngs_modulus,
    assign_youngs_modulus_v3
)
from analysis.material.fluctuations import add_boundary_fluctuations
from analysis.patterning.regions import (
    circle_six_arms_region,
    square_X_region
)
from analysis.patterning.complex import whole_peanut_region
from analysis.energy_check import test_fun_total_energy_grad_hess_thermal
from assembly.assemblers import (
    ElasticGHEdges, 
    ElasticGHEdgesCoupled, 
    ElasticGHEdgesCoupledThermal, 
    test_ElasticGHEdges, 
    FDM_ElasticGHEdges, 
    FDM_ElasticGHEdgesCoupled, 
    FDM_ElasticGHEdgesCoupledThermal)
from solver.boundary_conditions import (
    BoundaryConditions3D,
    fun_BC_3D_hold_center,
    fun_BC_peanut
)
from solver.time_stepper import timeStepper3D_static, record_step
from analysis.bending_model.geometry import getTheta

start = time.perf_counter()

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

    if args.mesh is not None:        iMesh = args.mesh
    if args.eps_thermal is not None: eps_thermal = args.eps_thermal
    if args.do_print:                iPrint = 0
    if args.do_plot:                 iPlot = 0
    if args.do_test:                 iTest = 0
    if args.gravity is not None:     iGravity = args.gravity
    if args.fluctuate is not None:   iFluc = args.fluctuate

    print("CLI overrides:", {k: v for k, v in vars(args).items() if v is not None and v is not False})

mesh_files = {
    1: '../../data/mesh_python_circle_970nodes_scale100mm.txt',
    2: '../../data/mesh_rectangle_scaled_1215nodes_scale100mm_155mm.txt',
    3: '../../data/mesh_python_square_388nodes_scale100mm.txt',
}

if iMesh == 1:
    Yratio         = 1.0   
    OuterR         = 0.05 
    Pattern_center = [0.0, 0.0]
    StripeWidth    = 0.0059   
    StripeLength   = 0.0441 
    Stripe_r       = 0.0029
    
if iMesh == 2:
    Yratio         = 1.0
    OuterR         = 0.05
    delta_shape    = 0.02787
    n_spokes       = 6  
    star_radius    = 0.044595,
    star_thickness = 0.002787,
    beam_thickness = 0.002787
    
if iMesh == 3:
    Yratio         = 1.0
    OuterR         = 0.05
    Pattern_center = [0.05, 0.05]
    StripeWidth    = 0.0059
    StripeLength   = 0.0707
    Stripe_r       = 0.0006
    
    
# Magnitude percentage of fluctuation
epsilon_th_fluctuation = 2.0
eps_thermal_min = 0
eps_thermal_max = eps_thermal

Ysoft = 1.0e6
Yhard = 3.0e6

FactorKs = 10.0
FactorKb = 1.0 # Let Kb get smaller than geometry defined.



# lk: mean length
# h1: thickness of shrinky dink layer
# h2: thickness of PLA layer
PARAMS = { # lk, h1, h2
    1: (0.0032, 0.3e-3, 0.7e-3), 
    2: (0.0040, 0.3e-3, 0.6e-3),
    3: (0.0058, 0.3e-3, 1.0e-3),
}

try:
    lk, h1, h2 = PARAMS[iMesh]
except KeyError:
    raise ValueError(f"Unsupported mesh index: {iMesh!r}") from None

b1   = 1.0
D_1  = Ysoft * b1 * (h1 ** 3) / 12.0;
D_12 = bilayer_flexural_rigidity(h1, h2, Ysoft, Yhard)
print(f"Shrinky dink: D_1       = {D_1:.8f} per unit width")
print(f"Bilayer composite: D_12 = {D_12:.8f} per unit width")

kb1  = (2.0 / np.sqrt(3.0)) * D_1
kb12 = (2.0 / np.sqrt(3.0)) * D_12
print(f"kb1        = {kb1:.8f} N·m")
print(f"kb12       = {kb12:.8f} N·m")
print(f"kb12/kb1   = {kb12/kb1:.8f} N·m")


ks1  = Ysoft * h1 * (lk ** 2) * np.sqrt(3) / 2.0
ks2  = Yhard * h2 * (lk ** 2) * np.sqrt(3) / 2.0
ks12 = ks1 + ks2
print(f"ks1  = {ks1:.8f} N·m")
print(f"ks2  = {ks2:.8f} N·m")
print(f"ks12 = {ks12:.8f} N·m")

mesh_file = mesh_files[iMesh]
X0_4columns, ConnectivityMatrix_line, Triangles = load_mesh(mesh_file)

print("nodeXYZ:\n", X0_4columns)
print("Connectivity:\n", ConnectivityMatrix_line)
print("Triangles:\n", Triangles)

NP_total   = X0_4columns.shape[0]
Nedges     = ConnectivityMatrix_line.shape[0]
Ntriangles = Triangles.shape[0]

print(f"Number of nodes:     {NP_total}")
print(f"Number of edges:     {Nedges}")
print(f"Number of triangles: {Ntriangles}")


print("Reference edge lengths:")
L0 = fun_edge_lengths(X0_4columns, ConnectivityMatrix_line)
# for i, length in enumerate(L0):
#    print(f"L0_{i} = {length:.6f}")

# Convert to a 1D array of length N * 3 with row‐major order.
X0 = X0_4columns[:,1:4].ravel()
print("Vector X0 =", X0)

Ndofs = X0.shape
Ndofs = Ndofs[0]
print("Ndofs =", Ndofs)

# 2D plots for reference configuration
fig, ax = new_fig(1)
ax.triplot(
    X0_4columns[:, 1],   # x coords
    X0_4columns[:, 2],   # y coords
    (Triangles[:, 1:4]), # zero‑based triangles
    color='blue'
)

ax.set_aspect('equal')
ax.set_title('Triangle Mesh')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

if iPlot:
    # Figure 2: nodes + triangle IDs
    fig, ax = new_fig(2)
    # Draw triangles
    ax.triplot(
        X0_4columns[:, 1],
        X0_4columns[:, 2],
        (Triangles[:, 1:4]),
        color='blue'
    )
    
    # Annotate triangle IDs
    for tri in Triangles:
        tid      = int(tri[0])
        idxs     = tri[1:4].astype(int)
        centroid = X0_4columns[idxs, 1:3].mean(axis=0)
        ax.text(
            centroid[0], 
            centroid[1],
            str(tid),
            fontsize=12, 
            color='red',
            ha='center', 
            va='center'
        )
    
    # Plot & label nodes
    for node in X0_4columns:
        nid = int(node[0])
        x, y = node[1], node[2]
        ax.scatter(x, y, color='pink', s=20)
        ax.text(x, 
                y, 
                str(nid), 
                fontsize=12, 
                color='black',
                ha='center', 
                va='center')
    
    ax.set_aspect('equal')
    ax.set_title('Nodes & Triangle IDs')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
    # Plot nodes and edge IDs
    fig, ax = new_fig(3)
    ax.set_aspect('equal')
    
    # Draw edges
    for edge in ConnectivityMatrix_line:
        eid    = int(edge[0])
        n1, n2 = int(edge[1]), int(edge[2])
        x0, y0 = X0_4columns[n1, 1], X0_4columns[n1, 2]
        x1, y1 = X0_4columns[n2, 1], X0_4columns[n2, 2]
        ax.plot([x0, x1], [y0, y1], color='gray', linewidth=1)
    
        # Label edge at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, 
                my, 
                str(eid), 
                fontsize=15, 
                color='blue',
                ha='center', 
                va='center')
    
    # Plot & label nodes
    for node in X0_4columns:
        nid = int(node[0])
        x, y = node[1], node[2]
        ax.scatter(x, 
                   y, 
                   color='pink', 
                   s=20)
        ax.text(x, 
                y, 
                str(nid), fontsize=15, 
                color='black',
                ha='center', 
                va='center')
    
    ax.set_title('Nodes & Edges IDs)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
    # Plot nodes and edge IDs for a slender beam  
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_aspect('equal')
    
    # Draw edges
    for eid, n1, n2 in ConnectivityMatrix_line.astype(int):
        x0, y0 = X0_4columns[n1,1], X0_4columns[n1,2]
        x1, y1 = X0_4columns[n2,1], X0_4columns[n2,2]
        ax.plot([x0, x1], [y0, y1], color='gray', linewidth=1)
        ax.text((x0+x1)/2, (y0+y1)/2, str(eid), fontsize=6,
                color='blue', ha='center', va='center')
    
    # Draw nodes (Fix unpacking)
    for node in X0_4columns:
        nid = int(node[0])
        x, y = node[1], node[2]
        ax.scatter(x, 
                   y, 
                   color='pink', 
                   s=10)
        ax.text(x, 
                y, 
                str(nid), 
                fontsize=6,
                color='black', 
                ha='center', 
                va='center')
    
    # Zoom to x in [0,1]
    ax.set_xlim(0, 1)
    ymin, ymax = X0_4columns[:,2].min(), X0_4columns[:,2].max()
    ax.set_ylim(ymin - 0.01, ymax + 0.01)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Nodes & Edges IDs: 0 ≤ x ≤ 1')
    plt.tight_layout()
    plt.show()

# 3D triangle‐wireframe 
fig, ax = new_fig(4, projection='3d')

for tri in Triangles:
    # Extract the 1‑based node IDs and convert to 0‑based indices
    i1, i2, i3 = int(tri[1]) , int(tri[2]) , int(tri[3]) 

    # Grab their x,y,z from X0_4columns
    p0 = X0_4columns[i1, 1:4]
    p1 = X0_4columns[i2, 1:4]
    p2 = X0_4columns[i3, 1:4]

    # Draw the triangle edges
    for a, b in ((p0, p1), (p1, p2), (p2, p0)):
        ax.plot([a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                color='blue', 
                linewidth=0.5)

ax.set_title('3D Triangle Mesh')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.auto_scale_xyz(X0_4columns[:,1],
                  X0_4columns[:,2],
                  X0_4columns[:,3])
plt.show()


if iPlot:
    # 3D nodes and triangle IDs
    fig, ax = new_fig(5, projection='3d')

    # Lightly draw every triangle again
    for tri in Triangles:
        i1, i2, i3 = int(tri[1]), int(tri[2]), int(tri[3])
        p0 = X0_4columns[i1, 1:4]
        p1 = X0_4columns[i2, 1:4]
        p2 = X0_4columns[i3, 1:4]

        for a, b in ((p0, p1), (p1, p2), (p2, p0)):
            ax.plot([a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color='gray', 
                    linewidth=1)
    
    # Annotate triangle IDs at centroids
    for tri in Triangles:
        tid = int(tri[0])
        idxs = [int(tri[j]) for j in (1,2,3)]
        pts = X0_4columns[idxs, 1:4]
        centroid = pts.mean(axis=0)
        ax.text(*centroid, str(tid), color='red')
    
    # Plot & label nodes
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
    
    
    # 3D nodes and edge IDs
    fig, ax = new_fig(6, projection='3d')

    for edge in ConnectivityMatrix_line:
        eid = int(edge[0])
        i0, i1 = int(edge[1]), int(edge[2])
        p0 = X0_4columns[i0, 1:4]
        p1 = X0_4columns[i1, 1:4]
    
        # Draw the edge
        ax.plot([p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color='gray', 
                linewidth=1)
    
        # Label at midpoint
        mid = (p0 + p1) / 2
        ax.text(*mid, str(eid), color='blue')
    
    # Plot & label nodes
    for node in X0_4columns:
        nid = int(node[0])
        x, y, z = node[1], node[2], node[3]
        ax.scatter(x, y, z, color='pink', s=20)
        ax.text(x, y, z, str(nid), color='black')
    
    ax.set_title('3D Nodes & Edge IDs')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.auto_scale_xyz(X0_4columns[:,1],
                      X0_4columns[:,2],
                      X0_4columns[:,3])
    plt.show()

node_xyz = X0_4columns[:,1:4]   # Shape (Nnodes,3)

xs = node_xyz[:,0]
ys = node_xyz[:,1]

x_center = 0.5*(xs.min() + xs.max())
y_center = 0.5*(ys.min() + ys.max())

print(f"Domain center: x={x_center:.4f}, y={y_center:.4f}")

ellipse_axes = ((xs.max() - xs.min()) / 2,
                (ys.max() - ys.min()) / 2)


Pattern_center = (x_center, y_center)

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
    else:  
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
            amp         = eps_thermal * 5,   # Max fluctuation
            n_waves     = 0.5,               # Sin‑cycles along each side
            decay_width = 0.03,              # Fade‑into‑interior width
        )

    Y_array = assign_youngs_modulus_v3(
        node_xyz,
        ConnectivityMatrix_line,
        region_fn      = region_fn,
        circle_center  = Pattern_center,
        circle_radius  = OuterR,
        Ysoft          = Ysoft,
        Yhard          = Yhard,
        Yratio         = Yratio,
        inside         = False,
        x_thresh_left  = 0.10,
        x_thresh_right = 0.185,
        hard_factor    = 1.0,
    )
else:
    raise ValueError(f"Unsupported mesh index for thermal/Y assignment: {iMesh!r}")

plot_thermal_strain_edges(
    node_coords  = node_xyz,
    connectivity = ConnectivityMatrix_line,
    epsilon_th   = eps_th_vector,
    title        = "thermal axial strain",
    cmap         = 'coolwarm' 
)
    
plot_thermal_strain_edges(
    node_coords  = node_xyz,
    connectivity = ConnectivityMatrix_line,
    epsilon_th   = Y_array,
    title        = "Young's modulus",
    cmap         = 'coolwarm'
)

plot_thermal_strain_edges_CustomRange(
    node_coords  = node_xyz,
    connectivity = ConnectivityMatrix_line,
    epsilon_th   = Y_array,
    title        = "Young's modulus (by Yhard)",
    cmap         = 'coolwarm',    
    vmin         = Yratio*Yhard,
    vmax         = Yhard)

plot_thermal_strain_edges_CustomRange(
    node_coords  = node_xyz,
    connectivity = ConnectivityMatrix_line,
    epsilon_th   = Y_array,
    title        = "Young's modulus (by Ysoft)",
    cmap         = 'coolwarm',    
    vmin         = Yratio*Ysoft,
    vmax         = Ysoft)

ks_array = np.where(Y_array == Yhard, ks12,
           np.where(Y_array == Ysoft, ks1, 0.0))


# Build a flat list of all undirected edges from the triangle list
tri_edges = []
for _, v1, v2, v3 in Triangles.astype(int):
    tri_edges += [
        tuple(sorted((v1, v2))),
        tuple(sorted((v2, v3))),
        tuple(sorted((v3, v1))),
    ]

# Count how many times each edge appears
edge_counts = Counter(tri_edges)

# Keep only those edges that appear exactly twice (interior edges = hinges)
hinge_keys = {edge for edge, cnt in edge_counts.items() if cnt == 2}

# Extract their IDs from your connectivity table
hinge_edges = []
for eid, n0, n1 in ConnectivityMatrix_line.astype(int):
    if tuple(sorted((n0, n1))) in hinge_keys:
        hinge_edges.append(eid)

hinge_edges = np.array(hinge_edges, dtype=int)
print("Hinge edge IDs:", hinge_edges)

all_edges = np.arange(Nedges, dtype=int)

# Boundary edges are those in all_edges but not in hinge_edges
boundary_edges = np.setdiff1d(all_edges, hinge_edges)

print("All edges:     ", all_edges)
print("Hinge edges:   ", hinge_edges)
print("Boundary edges:", boundary_edges)

edge_to_opps = defaultdict(list)

for _, v1, v2, v3 in Triangles.astype(int):
    verts = (v1, v2, v3)
    # For each of the 3 edges of this triangle:
    for a, b in combinations(verts, 2):
        key = tuple(sorted((a, b))) # Undirected edge
        # The “opposite” vertex is the one not in (a,b)
        opp = next(v for v in verts if v not in key)
        edge_to_opps[key].append(opp)

# Collect the four nodes for each hinge edgeID
hinge_quads = []  # Will hold rows [edgeID, n0, n1, oppA, oppB]

# Build a quick lookup from edgeID -> (n0, n1)
edgeid_to_nodes = {int(eid): (int(n0), int(n1))
                   for eid, n0, n1 in ConnectivityMatrix_line.astype(int)}

for eid in hinge_edges:
    n0, n1 = edgeid_to_nodes[eid]
    key = tuple(sorted((n0, n1)))
    oppA, oppB = edge_to_opps[key]   # The two “third” vertices
    hinge_quads.append([eid, n0, n1, oppA, oppB])

# Convert to a NumPy array
HingeQuads = np.array(hinge_quads, dtype=int)

print("edgeID, node0, node1, node2, node3")
print(HingeQuads)

HingeQuads

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

if iTest ==1:
    test_ElasticGHEdges(
        energy_choice    = 3,
        theta_bar        = 0.0,
        Nedges           = Nedges,
        NP_total         = NP_total,
        Ndofs            = Ndofs,
        ConnectivityMatrix_line     = ConnectivityMatrix_line,
        L0           = L0,
        X0               = X0,
        iPrint           = iPrint,
        HingeQuads_order = HingeQuads_order)
    

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
        theta_bar        = 0.0,
        plt = plt)



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
        eps=1e-6,
        plt=plt)



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
        eps=1e-6,
        plt=plt)

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
    test_fun_total_energy_grad_hess_thermal(plt, elastic_model1, q1, iplot=1)


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
sio.savemat('../../output/output_deps_thermal_WithGravity.mat', mdict, do_compression=True)
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

mdict = {
    'X0_4columns':             X0_4columns,
    'ConnectivityMatrix_line': ConnectivityMatrix_line,
    'Triangles':               Triangles,
    'NP_total':                NP_total,
    'Nedges':                  Nedges,
    'Ntriangles':              Ntriangles,
    'Ndofs':                   Ndofs,
    'X0':                      X0,
    'L0':                      L0,
    'ks_array':                ks_array,
    'kb_array':                kb_array,
    'Q_history':               Q_history,
    'R_history':               R_history,
    'strain_history':          strain_history,
    'stress_history':          stress_history,
    'HingeQuads_order':        HingeQuads_order,
    'theta_history':           theta_history,
    'time_log':                time_log,
    'step_log':                step_log,
    'q_old':                   q_old,
    'fixedNodes':              fixedNodes,
    'extra_records':           extra_records
}

# save data with compression
sio.savemat('../../output/output_deps_thermal_NoG.mat', mdict, do_compression=True)
print("Data saved at ", ", ".join(mdict.keys()))


end = time.perf_counter()
print(f"Elapsed time: {end - start:.4f} s")