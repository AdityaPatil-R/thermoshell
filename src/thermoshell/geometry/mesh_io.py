import numpy as np

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