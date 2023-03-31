import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# pts = points_M[safest_run == 1]
#
# times = np.unique(pts[:, 3])
# times_diff = times[1:] - times[:-1]
# times_diff = np.insert(times_diff, 0, 1.)
# clusters = np.zeros(pts.shape[0])
#
# class_idx = 1
# for t_idx in range(len(times_diff)):
#
#     if times_diff[t_idx] != 1.:
#         class_idx +=1
#
#     t = times[t_idx]
#     clusters[pts[:,3] == t] = class_idx
#
# for idx in np.unique(clusters):
#     object_pts = pts[clusters==idx]
#
#     split_pcl_list = [object_pts[object_pts[:,3] == t] for t in sorted(np.unique(object_pts[:,3]))]
#
#     # for t in range(len(split_pcl_list) - 1):
#     #     chamfer = pcu.chamfer_distance(split_pcl_list[t+1][:,:3], split_pcl_list[t][:,:3])
#     #     print(chamfer)
#
#     if idx == 1:break
#
# last_pts = split_pcl_list[-1]
#
# from datatools.structures.geometry import min_square_by_pcl, chamfer_distance, point_distance_from_hull
# surr_mask = min_square_by_pcl(points_M, last_pts, extend_dist=3, return_mask=True)
#
# visualize_points3D(points_M[surr_mask])
#
# ground = fit_ground_plane_under_ego(points_M[surr_mask])
#
# visualize_points3D(points_M[surr_mask], ground)
#
# from scipy.spatial import ConvexHull
# x = split_pcl_list[-1][:,:3]
# y = split_pcl_list[-2][:,:3]
#
# both_times = np.concatenate((x,y))
#
# hull = ConvexHull(both_times[:, :2])
# hull_points = both_times[hull.vertices, :2]
#
# point_distance_from_hull(x, both_times, plot=True)
#
#
# final_metric = 1000
# max_pts = 0
# # what about the angle?
# for i in range(-10,16, 1):
#     for j in range(-10,16, 1):
#         if i == 0 and j == 0: continue
#
#         move_pts = both_times[:,:3] + np.array((i/10, j/10, 0))
#         criterion_pts = x
#         tmp_metric = np.linalg.norm(point_distance_from_hull(x, move_pts, plot=False))
#         if tmp_metric < final_metric:
#             final_metric = np.linalg.norm(point_distance_from_hull(x, move_pts, plot=True))
#
#         print(f"Iter: {i * 26 + j}/{26 * 26} Frob norm {tmp_metric:.2f}")

# # beware of direction
# from sklearn.neighbors import NearestNeighbors
# x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(x)
# min_y_to_x = x_nn.kneighbors(y)[0]
# chamfer_dist = np.mean(min_y_to_x)
#


### CHAMFER MATCHING
# min_dist = 4
# min_idx = [0, 0]
# for i in range(20):
#     for j in range(20):
#         move_pts = last_pts[:,:3] + np.array((i/10, j/10, 0))
#         environ_pts = points_M[surr_mask][ground==False,:3]
#         tmp_min_dist = chamfer_distance(environ_pts, move_pts)
#
#         if tmp_min_dist < min_dist:
#             print(min_dist)
#             min_dist = tmp_min_dist
#             min_idx[0] = i
#             min_idx[1] = j

### KALMAN TRACKING
# from tracking.kalman_tracking import run_multi_object
#
# for idx in np.unique(clusters):
#     object_pts = pts[clusters==idx]
#     split_pcl_list = [object_pts[object_pts[:,3] == t] for t in sorted(np.unique(object_pts[:,3]))]
#     split_cluster_list = [t * np.ones(split_pcl_list[t].shape[0]) for t in range(len(split_pcl_list))]
#     track_poses = np.stack([poses[int(t)] for t in sorted(np.unique(object_pts[:,3]))])
#
#     run kalman separately
# tracking_data = run_multi_object('mos_ego', pcls=split_pcl_list, ids=split_cluster_list, poses=track_poses)
# print(tracking_data)
# visualize_points3D(object_pts, object_pts[:,3])

# visualize_points3D(points_M, safest_run)
