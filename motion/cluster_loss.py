import socket
import sys

import numpy as np
import torch
import open3d as o3d
from datasets.waymo.waymo import Waymo_Sequence
from datasets.visualizer import *

import torch

from datasets.paths import TMP_VIS

if socket.gethostname().startswith('Patrik') and False:
    import sys

    e = int(sys.argv[1])

    data_dict = np.load(TMP_VIS + f'/{e}_flow.npz', allow_pickle=True)

    freespace = data_dict['freespace']
    # freespace[:, 2] = freespace[:, 3]
    # visualize_multiple_pcls(*[data_dict['p_i'],  data_dict['p_i'][:,:3] + data_dict['flow'], data_dict['p_j']])

    visualize_flow3d(data_dict['p_i'], data_dict['p_j'], data_dict['flow'])

    # visualize_points3D(data_dict['p_i'], data_dict['p_i'][:, 2] > 0)
    # visualize_points3D(data_dict['p_i'], data_dict['loss'])
    sys.exit('Done')




def o3d_icp(p_i, p_j, threshold=0.01):
    p_i_centroid = p_i.mean(0)
    p_j_centroid = p_j.mean(0)

    p_i_centered = p_i - p_i_centroid
    p_j_centered = p_j - p_j_centroid

    pcd_i = o3d.geometry.PointCloud()
    pcd_i.points = o3d.utility.Vector3dVector(p_i_centered)

    pcd_j = o3d.geometry.PointCloud()
    pcd_j.points = o3d.utility.Vector3dVector(p_j_centered)

    t = p_j_centroid - p_i_centroid
    trans_init = np.eye(4)
    trans_init[:3, -1] = t

    reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_i, pcd_j, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    icp_transformation = reg_p2p.transformation
    # print(icp_transformation)
    return icp_transformation


def transformation_residuals(x1, x2, R, t):
    """
    Computer the pointwise residuals based on the estimated transformation paramaters

    Args:
        x1  (torch array): points of the first point cloud [b,n,3]
        x2  (torch array): points of the second point cloud [b,n,3]
        R   (torch array): estimated rotation matrice [b,3,3]
        t   (torch array): estimated translation vectors [b,3,1]
    Returns:
        res (torch array): pointwise residuals (Eucledean distance) [b,n,1]
    """
    x2_reconstruct = torch.matmul(R, x1.transpose(1, 2)) + t

    res = torch.norm(x2_reconstruct.transpose(1, 2) - x2, dim=2)

    return res

def kabsch_transformation_estimation(x1, x2, weights=None, normalize_w = True, eps = 1e-7, best_k = 0, w_threshold = 0, compute_residuals = False):
    """
    Torch differentiable implementation of the weighted Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm). Based on the correspondences and weights calculates
    the optimal rotation matrix in the sense of the Frobenius norm (RMSD), based on the estimated rotation matrix it then estimates the translation vector hence solving
    the Procrustes problem. This implementation supports batch inputs.
    Args:
        x1            (torch array): points of the first point cloud [b,n,3]
        x2            (torch array): correspondences for the PC1 established in the feature space [b,n,3]
        weights       (torch array): weights denoting if the coorespondence is an inlier (~1) or an outlier (~0) [b,n]
        normalize_w   (bool)       : flag for normalizing the weights to sum to 1
        best_k        (int)        : number of correspondences with highest weights to be used (if 0 all are used)
        w_threshold   (float)      : only use weights higher than this w_threshold (if 0 all are used)
    Returns:
        rot_matrices  (torch array): estimated rotation matrices [b,3,3]
        trans_vectors (torch array): estimated translation vectors [b,3,1]
        res           (torch array): pointwise residuals (Eucledean distance) [b,n]
        valid_gradient (bool): Flag denoting if the SVD computation converged (gradient is valid)
    """
    if weights is None:
        weights = torch.ones(x1.shape[0],x1.shape[1]).type_as(x1).to(x1.device)

    if normalize_w:
        sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
        weights = (weights/sum_weights)

    weights = weights.unsqueeze(2)

    if best_k > 0:
        indices = np.argpartition(weights.cpu().numpy(), -best_k, axis=1)[0,-best_k:,0]
        weights = weights[:,indices,:]
        x1 = x1[:,indices,:]
        x2 = x2[:,indices,:]

    if w_threshold > 0:
        weights[weights < w_threshold] = 0


    x1_mean = torch.matmul(weights.transpose(1,2), x1) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)
    x2_mean = torch.matmul(weights.transpose(1,2), x2) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)

    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean

    print(x1_mean, x2_mean)
    cov_mat = torch.matmul(x1_centered.transpose(1, 2),
                            (x2_centered * weights))

    try:
        u, s, v = torch.svd(cov_mat)

    except Exception as e:
        r = torch.eye(3,device=x1.device)
        r = r.repeat(x1_mean.shape[0],1,1)
        t = torch.zeros((x1_mean.shape[0],3,1), device=x1.device)

        res = transformation_residuals(x1, x2, r, t)

        return r, t, res, True

    tm_determinant = torch.det(torch.matmul(v.transpose(1, 2), u.transpose(1, 2)))

    determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0],2),device=x1.device), tm_determinant.unsqueeze(1)), 1))

    rotation_matrix = torch.matmul(v,torch.matmul(determinant_matrix,u.transpose(1,2)))

    # translation vector
    translation_matrix = x2_mean.transpose(1,2) - torch.matmul(rotation_matrix,x1_mean.transpose(1,2))

    # Residuals
    res = None
    if compute_residuals:
        res = transformation_residuals(x1, x2, rotation_matrix, translation_matrix)

    return rotation_matrix, translation_matrix, res, False


sequence = Waymo_Sequence(1)
# load data
frame = 0
p_i = torch.tensor(sequence.get_feature(idx=frame, name='lidar')[None, :,:3], dtype=torch.float)
p_j = torch.tensor(sequence.get_feature(idx=frame+1, name='lidar')[None, :,:3], dtype=torch.float)

p_i = p_i[p_i[..., 2] > 0.3]
p_j = p_j[p_j[..., 2] > 0.3]

local_pts = []
till_frame=25

x_max, x_min = 35, -35
y_max, y_min = 35, -35

for i in range(frame, frame+till_frame):
    p_i = sequence.get_feature(idx=i, name='lidar')[:,:4]
    p_i = p_i[(p_i[..., 0] > x_min) & (p_i[..., 0] < x_max) &
              (p_i[..., 1] > y_min) & (p_i[..., 1] < y_max) &
              (p_i[..., 2] > 0.3) & (p_i[..., 2] < 2.)]
    pose = sequence.get_feature(idx=i, name='pose')
    global_pts = sequence.pts_to_frame(p_i, pose)
    local_pts.append(np.insert(global_pts, 3, i, axis=1))

global_pts = np.concatenate(local_pts)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.2)
global_pts[:,3] *= 0.198

# first learn ego-motion self-supervised
# get road by variance
# then spatio-temporal clustering on point sequence
# then splitting clustering on the ids with same/higher epsilon (full sequence object)
# then remove outliers by size percentile for example and fit rigid transformation as weakly-supervised
dbscan_ids = dbscan.fit_predict(global_pts[:, [0,1,2,3]])

visualize_points3D(global_pts[:, [0,1,2]], global_pts[:,3])
# visualize_points3D(global_pts[:, [0,1,2]], global_pts[:,4])
visualize_points3D(global_pts[:, [0,1,3]], dbscan_ids)

object_id = 149

object_pts = global_pts[dbscan_ids == object_id]

avail_t = np.unique(object_pts[:, 3])

split_pts = [object_pts[object_pts[:, 3] == t] for t in avail_t]

idx = 0
p_i = split_pts[idx][:,:3]
p_j = split_pts[idx + 1][:, :3]
subsample_idx = np.random.randint(0, p_j.shape[0], (p_i.shape[0], 1))[:, 0]
p_j = p_j[subsample_idx, :3]


# kabsch_transformation_estimation(p_i[..., :3], p_j[..., :3])

p_i_centroid = p_i.mean(0)
p_j_centroid = p_j.mean(0)

p_i_centered = p_i - p_i_centroid
p_j_centered = p_j - p_j_centroid

t = p_j_centroid - p_i_centroid

h = p_i_centered.T @ p_j_centered   # mapping, to which to map
u, s, vt = np.linalg.svd(h)
v = vt.T

d = np.linalg.det(v @ u.T)
e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

r = v @ e @ u.T

object_rigid_flow = np.tile(t, (p_i.shape[0], 1))
object_rigid_flow[:, 2] = 0 # jen pro visualizaci

best_object_rigid_flow = object_rigid_flow
visualize_flow3d(p_i, p_j, object_rigid_flow)
metric = np.inf

from scipy.spatial.distance import cdist

for trans_scale in np.linspace(-1,1,21):
    distance_matrix = cdist(p_i + t + t * (trans_scale, trans_scale, 0), p_j)
    p_i_min_NN_dist = np.min(distance_matrix, axis=1)
    curr_result = np.sum(p_i_min_NN_dist)

    if curr_result < metric:

        metric = curr_result
        best_object_rigid_flow = object_rigid_flow + t + t * (trans_scale, trans_scale, 0)
        print(metric, trans_scale)

visualize_flow3d(p_i, p_j, best_object_rigid_flow)

# rollin in the object
accum_pts = [split_pts[0][:,:3] - split_pts[0][:, :3].mean(0)]
trajectory = []
for i in range(len(split_pts) - 1):
    p_i = np.concatenate(accum_pts)
    p_i -= p_i.mean(0)
    p_j = split_pts[i+1][:, :3]
    p_j -= p_j.mean(0)
    # next one is source, we fit new to already accumulated
    icp_transform = o3d_icp(p_j, p_i, threshold=0.02)

    trans_p_j = np.insert(p_j, obj=3, values=1, axis=1)
    missing_p_j = trans_p_j @ icp_transform.T
    accum_pts.append(missing_p_j[:,:3])

    trajectory.append(icp_transform[:3, -1][None, :])

all_list = accum_pts + trajectory
visualize_multiple_pcls(*all_list)
# visualizer_transform(p_i, p_j, icp_transformation)
