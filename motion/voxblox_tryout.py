from voxblox import (
    BaseTsdfIntegrator,
    FastTsdfIntegrator,
    MergedTsdfIntegrator,
    SimpleTsdfIntegrator,
)

from my_datasets.delft.drone import Delft_Sequence
import glob
import numpy as np

sequence = Delft_Sequence(0)

# files = sorted(glob.glob('/home/patrik/patrik_data/drone/hrnet_pts/*.npy'))
files = sorted(glob.glob('/home/patrik/patrik_data/drone/hrnet_pts_filtered/*.npy'))

voxel_size = 0.5
sdf_trunc = 3 * voxel_size

voxblox_integrator = SimpleTsdfIntegrator(voxel_size, sdf_trunc)

for i in range(len(sequence)):
    print(i)
    pts = np.load(files[i])
    pose = np.eye(4)
    voxblox_integrator.integrate(pts[:,:3].astype(np.float64), pose)

    if i == 80: break


from my_datasets.visualizer import visualize_points3D
visualize_points3D(pts[:,:3], pts[:,3:6])

import open3d as o3d

# Get the output mesh
vertices, triangles = voxblox_integrator.extract_triangle_mesh()
mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices),
    o3d.utility.Vector3iVector(triangles),
)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
