import numpy as np
from typing import Tuple, List

# --- Mesh File Definitions ---
# This dictionary maps the integer mesh_id (used in core/params.py) to its file path.
mesh_files = {
    1: 'mesh_python_circle_970nodes_scale100mm.txt',
    2: 'mesh_rectangle_scaled_1215nodes_scale100mm_155mm.txt',
    3: 'mesh_python_square_388nodes_scale100mm.txt',
}


def load_mesh(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads nodal coordinates, edge connectivity, and triangle connectivity 
    from a custom mesh text file.
    
    The file format uses marker lines (*...) to denote sections:
    *shellNodes: [ID, X, Y, Z]
    *FaceNodes: [EID, N1, N2, N3]
    *Edges: [EID, N1, N2]
    
    Parameters:
        filename (str): The path to the mesh file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            nodeXYZ (Nnodes, 4): [nodeID, x, y, z]
            Connectivity (Nedges, 3): [edgeID, node_i, node_j]
            Triangles (Ntris, 4): [triID, node_1, node_2, node_3]
    """
    nodeXYZ: List[List[float]] = []
    Triangles: List[List[int]] = []
    Connectivity: List[List[int]] = []

    section = None
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Detect new section
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

                # Parse a data line
                parts = [p.strip() for p in line.split(',')]
                if not parts or parts[0] == '':
                    continue
                
                # --- NODE Section ---
                if section == 'nodes' and len(parts) >= 4:
                    # node ID, x, y, z
                    nid = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    nodeXYZ.append([nid, x, y, z])

                # --- TRIANGLE Section ---
                elif section == 'triangles' and len(parts) >= 4:
                    # element ID, node1, node2, node3 (0-based indices)
                    eid = int(parts[0])
                    n1, n2, n3 = map(int, parts[1:4])
                    Triangles.append([eid, n1, n2, n3])

                # --- EDGE Section ---
                elif section == 'edges' and len(parts) >= 3:
                    # edge ID, node1, node2 (0-based indices)
                    eid = int(parts[0])
                    n1, n2 = map(int, parts[1:3])
                    Connectivity.append([eid, n1, n2])

    except FileNotFoundError:
        print(f"Error: Mesh file not found at {filename}")
        raise
    except Exception as e:
        print(f"Error parsing mesh file {filename}: {e}")
        raise

    # Convert to NumPy arrays
    nodeXYZ      = np.array(nodeXYZ, dtype=float)
    Triangles    = np.array(Triangles, dtype=int)
    Connectivity = np.array(Connectivity, dtype=int)

    return nodeXYZ, Connectivity, Triangles
