import os

import numpy as np
import torch
from tqdm import tqdm
from my_datasets.delft.drone import Delft_Sequence
from my_datasets.visualizer import *
from my_datasets.structures.rgbd import plot_realsense
from scipy.spatial.transform.rotation import Rotation


from voxblox import (
    BaseTsdfIntegrator,
    FastTsdfIntegrator,
    MergedTsdfIntegrator,
    SimpleTsdfIntegrator,
)

# todo
# codes
# transform poses for voxblox
# icp on data, kabsch on data
# onnx format of networks
# training in format for Max


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T


    # print(f" Average error: {rmsd} m")

    return R, t.squeeze()


sequence = Delft_Sequence(0)
# print(sequence.sequence_path)
print(sequence.sequence_dict)

import open3d as o3d
import copy

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])


# colored pointcloud registration
# This is implementation of following paper
# J. Park, Q.-Y. Zhou, V. Koltun,
# Colored Point Cloud Registration Revisited, ICCV 2017

def color_icp_registration(source, target):

    voxel_radius = [1., 1., 1.]
    max_iter = [1000, 500, 340]
    current_transformation = np.identity(4)
    # print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([iter, radius, scale])

        # print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        # print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        # print(result_icp)

    return current_transformation



pcd_paths = sorted(glob.glob(f'/home/patrik/patrik_data/drone/kiss_icp/*.pcd'))
frame = 0

source = o3d.io.read_point_cloud(pcd_paths[frame])
target = o3d.io.read_point_cloud(pcd_paths[frame+1])

current_transformation = color_icp_registration(source, target)

# draw_registration_result_original_color(source, target,
#                                         current_transformation)


paths = sorted(glob.glob(f'/home/patrik/patrik_data/drone/sequences/cor1/pts/*'))

for frame in range(len(paths)):

    new_path = f'/home/patrik/patrik_data/drone/kiss_icp/{frame:06}.pcd'
    np.load(paths[frame])
    # np_pcd = sequence.get_feature(frame, 'sync_pts')
    # pts = sequence.get_feature(frame, 'sync_pts')

    pts[:, [0, 2]] = pts[:, [2, 0]]
    pts[:, [1, 2]] = pts[:, [2, 1]]
    pts[:, 2] = - pts[:, 2]
    pts[:, 1] = - pts[:, 1]

    # filter out longer distances
    dist_thresh = 5
    dist_mask = pts[:, 0] < dist_thresh

    pts = pts[dist_mask]

    pts[:, 3:6] /= 255  # rgb to float

    np_pcd = pts

    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd[:,:3])
    pcd.colors = v3d(np_pcd[:,3:])

    o3d.io.write_point_cloud(new_path, pcd)
    print(new_path)

    # if frame == 200:
    #     break

from tqdm import tqdm
for frame in tqdm(range(len(sequence))):

    # local
    source = o3d.io.read_point_cloud(pcd_paths[frame])
    target = o3d.io.read_point_cloud(pcd_paths[frame+1])
    threshold = 0.02

    # transformation is between frames, not global. You need to multiple along the sequence
    init_pose1 = sequence.get_feature(frame, 'sync_pose')
    init_pose2 = sequence.get_feature(frame+1, 'sync_pose')

    trans_init = np.linalg.inv(init_pose1) @ init_pose2
    # trans_init = np.eye(4)
    # break

    # draw_registration_result(source, target, trans_init)

    transform_matrix = color_icp_registration(source, target)

    # print(f"Generating pose for frame {frame}")


    sequence.store_feature(transform_matrix, frame, name='sync_open3d_icp_pose')

    # multiply by sequence

    if frame == 200:
        break

new_pose_list = [sequence.get_feature(i, name='sync_open3d_icp_pose') for i in range(200)]

# last_res_pose = reg_p2p.transformation
# pose = np.eye(4)
global_pts_list = []
pose_pts = []
# len(new_pose_list)
for frame in range(50,100):
    print(frame)
    pts = sequence.get_feature(frame, name='sync_pts')

    # pts[:, [0, 2]] = pts[:, [2, 0]]
    # pts[:, [1, 2]] = pts[:, [2, 1]]
    # pts[:, 2] = - pts[:, 2]
    # pts[:, 1] = - pts[:, 1]

    # filter out longer distances
    # dist_thresh = 5
    # dist_mask = pts[:, 0] < dist_thresh

    # pts = pts[dist_mask]

    pts[:, 3:6] /= 255  # rgb to float

    pose = np.eye(4)
    for i in range(frame):
        pose = np.dot(pose, new_pose_list[i])

    pose_pts.append(pose[:3,-1])
    global_pts_list.append(sequence.pts_to_frame(pts, pose))

pose_pts = np.stack(pose_pts)
visualize_points3D(pose_pts)

vis_pts = np.concatenate(global_pts_list)
visualize_points3D(vis_pts[:,:3], vis_pts[:,3:6])

poses = np.stack([sequence.get_feature(idx, 'sync_pose') for idx in range(len(sequence))])
pose_pts2 = poses[:,:3,-1]

visualize_multiple_pcls(pose_pts, pose_pts2)
# rot_offset = Rotation.from_euler('xyz', [0,0,-np.pi/2], degrees=False).as_matrix()
#
# poses[:,:3,:3] = poses[:, :3, :3] @ rot_offset


vis_list = []
for i in range(0, 1, 1):
    pose = poses[i]
    pts = sequence.get_feature(i, 'sync_pts')

    pts[:, [0,2]] = pts[:,[2,0]]
    pts[:, [1, 2]] = pts[:, [2,1]]
    pts[:, 2] = - pts[:, 2]
    pts[:, 1] = - pts[:, 1]

    # filter out longer distances
    dist_thresh = 5
    dist_mask = pts[:, 0] < dist_thresh

    pts = pts[dist_mask]

    pts[:, 3:6] /= 255 # rgb to float
    global_pts = sequence.pts_to_frame(pts, pose)

    vis_list.append(global_pts)

all_pts = np.concatenate(vis_list)

print(all_pts.shape)

if pose_pts.shape[1] == 3:
    pose_pts = np.insert(pose_pts, 3, 1, axis=1)
    pose_pts = np.insert(pose_pts, 4, 0, axis=1)
    pose_pts = np.insert(pose_pts, 5, 0, axis=1)

# visualize_points3D(all_pts, all_pts[:,3:6], point_size=0.001)
rot_vecs = Rotation.from_matrix(poses[:,:3,:3]).as_rotvec()

# visualize_points3D(pose_pts, (rot_vecs[:,:3] + np.pi) / 2 / np.pi )

all_pts = np.concatenate((all_pts, pose_pts[:,:6]))

visualize_points3D(all_pts, all_pts[:,3:6], point_size=0.004)

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(4, 4), dpi=200)
# ax = fig.add_subplot(projection='3d')
# ax.scatter(pose_pts[:,0], pose_pts[:,1], pose_pts[:,2])
#
# ax.quiver(pose_pts[:, 0], pose_pts[:, 1], pose_pts[:,2], dx, dy, dz)
# ax.set_box_aspect((1, 1, 1))
# plt.axis('equal')
#
# plt.show()
# visualize_points3D(all_pts, all_pts[:,3:6], point_size=0.001)




# what has effect:
# 1) Calibration parameters
# 2) Openvins, but the pose is correct
# 3) creation of point cloud from depth


sys.exit('1')

pts_list = []
pose_list = []

from sklearn.neighbors import NearestNeighbors
from motion_supervision.motionflow import pytorch3d_ICP

def subsample_by_NN(pts, KNN=100, min_KNN_points_closer_thresh=0.15):

    nbrs = NearestNeighbors(n_neighbors=KNN, algorithm='ball_tree').fit(pts[:, :3])
    distances, indices = nbrs.kneighbors(pts[:, :3])

    density_mask = distances.max(1) < min_KNN_points_closer_thresh

    final_pts = pts[density_mask]

    return final_pts, density_mask

st_frame = 0
pts_list.append(sequence.get_feature(st_frame, 'sync_pts'))
from tqdm import tqdm

for frame in tqdm(range(st_frame, st_frame + 10, 1)):

    # pose = sequence.get_feature(frame, 'sync_pose')
    pose = poses[frame]
    pts1 = pts_list[st_frame]
    pts2 = sequence.get_feature(frame, 'sync_pts')

    subsampled_pts1, mask1 = subsample_by_NN(pts1, KNN=40, min_KNN_points_closer_thresh=0.05)
    subsampled_pts2, mask2 = subsample_by_NN(pts2, KNN=40, min_KNN_points_closer_thresh=0.05)

    pts1 = subsampled_pts1
    pts2 = subsampled_pts2

    min_nbr_pts = np.min((pts1.shape[0], pts2.shape[0]))
    rot_pts1 = torch.tensor(pts1[:min_nbr_pts])
    rot_pts2 = torch.tensor(pts2[:min_nbr_pts])

    # Density subsampling using NN


    # Kabsch
    rot, trans = find_rigid_alignment(rot_pts1[:, :6], rot_pts2[:, :6])

    if torch.cuda.is_available():
        transform, trans_pts = pytorch3d_ICP(rot_pts1[:,:3].cuda(), rot_pts2[:,:3].cuda(), verbose=True)
        transform = transform.cpu()
        trans_pts = trans_pts.cpu()

    new_trans = np.eye(4)
    new_trans[:3, :3] = rot[:3,:3].numpy()
    new_trans[:3, -1] = trans[:3].numpy()

    transformed_pts1 = sequence.pts_to_frame(pts1, new_trans)
    # zxy ---> xyz
    # pts[:, [0, 2]] = pts[:, [2, 0]]
    # pts[:, 2] = - pts[:, 2]


    global_pts = sequence.pts_to_frame(pts1, new_trans)
    pts_list.append(global_pts)
# Different realsense parameters? The projection is messed up
all_pts = np.concatenate(pts_list, axis=0)
visualize_points3D(all_pts, all_pts[:, 3:6], point_size=0.002)

from motion_supervision.motionflow import pytorch3d_ICP

# Kabsch algorithm
pts1 = pts_list[0]
pts2 = pts_list[1]

min_nbr_pts = np.min((pts1.shape[0], pts2.shape[0]))
rot_pts1 = torch.tensor(pts1[:min_nbr_pts])
rot_pts2 = torch.tensor(pts2[:min_nbr_pts])



# transform, trans_pts = pytorch3d_ICP(rot_pts1[:,3].unsqueeze(0), rot_pts2[:,3].unsqueeze(0), verbose=True)

rot, trans = find_rigid_alignment(rot_pts1[:,:3], rot_pts2[:,:3])

new_trans = np.eye(4)
new_trans[:3,:3] = rot.numpy()
new_trans[:3,-1] = trans.numpy()

transformed_pts1 = sequence.pts_to_frame(pts1, new_trans)

rgb_pts = np.concatenate([transformed_pts1, pts2], axis=0)

visualize_points3D(rgb_pts, rgb_pts[:, 3:6], point_size=0.002)
visualize_multiple_pcls(*[transformed_pts1, pts2])
visualize_multiple_pcls(*[pts1, pts2])

if pose_pts.shape[1] == 3:
    pose_pts = np.insert(pose_pts, 3, np.arange(0, len(pose_pts)), axis=1)
    pose_pts = np.insert(pose_pts, 4, np.arange(0, len(pose_pts)), axis=1)
    pose_pts = np.insert(pose_pts, 5, np.arange(0, len(pose_pts)), axis=1)


all_pts = np.concatenate((all_pts, pose_pts[:,:6]))

visualize_points3D(all_pts, all_pts[:,3:6], point_size=0.002)

# need some alignment in 3d domain? ICP?

# voxblox
# # Pick some parameters
# voxel_size = 0.1
# sdf_trunc = 3 * voxel_size
#
# # Run fusion pipeline
# tsdf_volume = SimpleTsdfIntegrator(voxel_size, sdf_trunc)
#
# import open3d as o3d
#
# for idx in range(len(pose_list)):
#     print(idx)
#     scan, pose = pts_list[idx][:,:3].astype(np.float64), pose_list[idx].astype(np.float64)
#     tsdf_volume.integrate(scan, pose)
#
# # Get the output mesh
# vertices, triangles = tsdf_volume.extract_triangle_mesh()
# mesh = o3d.geometry.TriangleMesh(
#     o3d.utility.Vector3dVector(vertices),
#     o3d.utility.Vector3iVector(triangles),
# )
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])



# pts_files = sorted(glob.glob('/home/patrik/patrik_data/drone/hrnet_pts/*.npy'))

# Filtration!
# for idx in range(len(pts_files)):
#
#     pts = np.load(pts_files[idx])
#     # visualize_points3D(pts, pts[:,3:6])
#
#     # DISTANCE THRESHOLD
#     max_dist = 5
#     dist_mask = np.linalg.norm(pts[:,:3], axis=1) < max_dist
#     pts = pts[dist_mask]
#
#     # Random Subsampling
#     ratio = 0.3
#     nbr_of_kept_pts = int(len(pts) * ratio)
#     kept_indices = np.random.choice(len(pts), nbr_of_kept_pts, replace=False)
#     pts = pts[kept_indices]
#     # visualize_points3D(pts, pts[:,3:6])
#
#     # HDBSCAN clustering subsampling
#     from hdbscan import HDBSCAN
#     labels = HDBSCAN(min_cluster_size=60, min_samples=50).fit_predict(pts[:,:6])
#     # visualize_points3D(pts, labels)
#     # visualize_points3D(pts, labels==-1)
#
#     final_pts = pts
#     final_pts = final_pts[labels !=-1]
#
#     # Density subsampling using NN
#     from sklearn.neighbors import NearestNeighbors
#     import numpy as np
#     KNN = 100
#     nbrs = NearestNeighbors(n_neighbors=KNN, algorithm='ball_tree').fit(final_pts[:,:3])
#     distances, indices = nbrs.kneighbors(final_pts[:,:3])
#
#     min_KNN_points_closer_thresh = 0.15
#     density_mask = distances.max(1) < min_KNN_points_closer_thresh
#
#     final_pts = final_pts[density_mask]
#
#     os.makedirs('/home/patrik/patrik_data/drone/hrnet_pts_filtered', exist_ok=True)
#     os.makedirs('/home/patrik/patrik_data/drone/hrnet_pts_filtered_rgb', exist_ok=True)
#
#     file_path = '/home/patrik/patrik_data/drone/hrnet_pts_filtered/' + str(idx).zfill(6) + '.npy'
#     np.save(file_path, final_pts)
#
#
#     # for visuals [y] and swapping axes [x,z]
#     final_pts[:, [0, 2]] = final_pts[:, [2, 0]]
#     final_pts[:, 2] = - final_pts[:, 2]
#     final_pts[:, 1] = - final_pts[:, 1]
#
#     for features, name in zip([ [3,4,5], [6, 7, 8], [9]], ['rgb', 'seg_rgb', 'confidence']):
#         if name in ['confidence']:
#             cb = True
#         else:
#             cb = False
#
#         save_path = file_path.replace('.npy', '.png').replace('hrnet_pts_filtered', 'hrnet_pts_filtered_' + name)
#         plot_realsense(final_pts, final_pts[:, features], save=save_path, cb=cb, vmin=0, vmax=1)
#         print(f"Storing {name} of frame {idx}")

    # img = plt.imread(file_path.replace('.npy', '.png').replace('hrnet_pts_filtered', 'hrnet_pts_filtered_rgb'))
    # plt.imshow(img)
    # plt.show()
    # break

# if done
# global_pts_files = sorted(glob.glob('/home/patrik/patrik_data/drone/hrnet_pts_filtered/*.npy'))
# global_pts_list = [np.insert(np.load(f), 11, t, axis=1) for t, f in enumerate(global_pts_files)]
# global_pts = np.concatenate(global_pts_list, axis=0)
#
# conf_thresh = 0.9
# conf_mask = global_pts[:,9] > conf_thresh
# human_mask = (global_pts[:,10] == 36) & (global_pts[:, 9] > conf_thresh)
# wall_mask = (global_pts[:,10] == 55) & (global_pts[:, 9] > conf_thresh)
# ground_mask = (global_pts[:,10] == 25) & (global_pts[:, 9] > conf_thresh)
# chair_mask = (global_pts[:,10] == 16) & (global_pts[:, 9] > conf_thresh)
#
#
#
# # visualize_points3D(global_pts, global_pts[:,6:9])
# visualize_points3D(global_pts[conf_mask], global_pts[conf_mask, 6:9])
# visualize_points3D(global_pts[conf_mask], global_pts[conf_mask, 10])
#
#
# vis_pts = global_pts[chair_mask | wall_mask | ground_mask | human_mask]
# label_mask = chair_mask * 1 + wall_mask * 2 + ground_mask * 3 + human_mask * 4
# label_mask = label_mask[label_mask != 0]
#
# visualize_points3D(vis_pts, label_mask)
#
# cur_mask = chair_mask
# visualize_points3D(global_pts[cur_mask], global_pts[cur_mask, 3:6])


# video of filtered pts
# for i in range(len(global_pts_files)):
#     print(i)
#     conf_pts = global_pts_list[i]
#
#     conf_pts[:, [0, 2]] = conf_pts[:, [2, 0]]
#     conf_pts[:, 2] = - conf_pts[:, 2]
#     conf_pts[:, 1] = - conf_pts[:, 1]
#
#     conf_pts = conf_pts[conf_pts[:,9] > conf_thresh]
#     save_path = global_pts_files[i].replace('.npy', '.png').replace('hrnet_pts_filtered', 'hrnet_pts_filtered_by_conf')
#     plot_realsense(conf_pts, conf_pts[:, 6:9], save=save_path, cb=False, vmin=0, vmax=1)
#
# # read label list
# label_file = open('config/hrnet_label.txt', 'r')
# lines = label_file.readlines()
# d_lines = {int(line.split(':')[0]) : line.split(':')[-1].strip() for line in lines}
# visualize_points3D(final_pts, final_pts[:,3:6]) # RGB
# visualize_points3D(final_pts, final_pts[:,6:9]) # seg labels in rgb
# visualize_points3D(final_pts, final_pts[:,9])   # confidence
# visualize_points3D(final_pts, final_pts[:,9] > 0.9) # kept points by confidence

