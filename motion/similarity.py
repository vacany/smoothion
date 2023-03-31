import numpy as np
from sklearn.cluster import DBSCAN

from my_datasets.waymo.waymo import Waymo_Sequence
from my_datasets.visualizer import *
import torch

import sys
import time

start_time = time.time()

# This is important script for later perhaps!

def cluster_prior_sequence_pts(to_cluster, cell_size, min_samples):
    dbscan = DBSCAN(eps=cell_size[0], min_samples=5)  # todo check
    cluster_labels = dbscan.fit_predict(
        to_cluster[:, [0, 1, 3]] * np.array((1, 1, cell_size[0] * 0.99)))  # take two times and scale it for clustering

    return cluster_labels

sequence = Waymo_Sequence(0)

frame = 0
nbr_times = 40 #len(sequence)



seq_pts = [np.insert(sequence.get_global_pts(frame+i, name='lidar')[:,:3], obj=3, values=i, axis=1) for i in range(frame, frame + nbr_times)]
seq_dyn = [sequence.get_feature(frame+i, name='visibility_prior') for i in range(frame, frame + nbr_times)]
seq_flow_label = [sequence.get_feature(frame+i, name='flow_label') for i in range(frame, frame + nbr_times)]
seq_ground_dynamic = [sequence.get_feature(frame+i, name='dynamic_label') for i in range(frame, frame + nbr_times)]

seq_pts = np.concatenate(seq_pts)
seq_dyn = np.concatenate(seq_dyn)
seq_flow_label = np.concatenate(seq_flow_label)
seq_ground_dynamic = np.concatenate(seq_ground_dynamic)




to_cluster = seq_pts[seq_dyn == 1].copy()
cluster_labels = cluster_prior_sequence_pts(to_cluster, cell_size=(0.1,0.1,0.1), min_samples=5)

tensor_pts = torch.tensor(to_cluster, requires_grad=True)
tensor_cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)

indices = tensor_cluster_labels.sort()[1]
u, c = torch.unique(tensor_cluster_labels, return_counts=True)

accum_c = [torch.sum(c[:i]) for i in range(len(c))]

sorted_pts = tensor_pts[indices, :4]
split_tensor = sorted_pts.tensor_split(list(accum_c)) # it is shifted I think

visualize_points3D(to_cluster[:, [0,1, 3]] * np.array((1,1,0.2)), cluster_labels)

cosine_sim = torch.nn.CosineSimilarity(dim=0)   # 1 for BS

cos_loss = []

# calculation of loss
# todo check idx == -1
for obj_idx in range(1, len(split_tensor)): # to prevent -1 index without points
    obj_pts = split_tensor[obj_idx]
    cos_loss.append(cosine_sim(obj_pts[:,:3], obj_pts[:,:3].mean(0)).mean())

loss = torch.mean(torch.stack(cos_loss))

# visualize_multiple_pcls(*list(split_tensor))


# person[:, 2] =
#
# visualize_points3D(person[:, [0,1,3]]  * np.array((1,1,0.2)), person[:, 3])

# calculate safe radius

from timespace.geometry import get_max_size
from scipy.spatial.distance import cdist


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


to_cluster_dynamic_prediction = np.zeros(cluster_labels.shape[0], dtype=bool)
minimal_velocity_estimate = - np.ones(cluster_labels.shape[0], dtype=float)

for object_idx in np.unique(cluster_labels):
    if object_idx == -1: continue

    print(object_idx)

    orig_mapping = np.argwhere(cluster_labels == object_idx)[:, 0]

    person = to_cluster[orig_mapping] # alias for arbitrary object ...

    r = 0.9 # set min r as min radius of smallest moving object - person
    available_times = np.unique(person[:, 3]).astype('int')

    min_nbr_pts_dist = 2


    for t in available_times:

        one_time = person[person[:,3] == t]

        if len(one_time) <= min_nbr_pts_dist:
            continue

        med = np.median(one_time[:,:2], axis=0)
        mean = np.mean(one_time[:,:2], axis=0)

        # sometimes the clusters can be coplanar
        try:
            curr_r = get_max_size(one_time[:,:2])
        except:
            continue

        if curr_r > r:
            r = curr_r

    # meds
    median_coors = [np.median(person[person[:,3] == t,:2], axis=0) for t in available_times]


    # dynamic criterion

    # serialized by time, take first dynamic time
    cross_distance = cdist(median_coors, median_coors)

    dynamic_times = np.zeros(available_times.shape, dtype=bool)
    dynamic_mask = np.zeros(person.shape[0], dtype=bool)
    # todo do this back and forth


    for row in range(len(median_coors)):

        cross_distance = cdist(median_coors, median_coors[row][None,:])[:,0]


        outside_idx = np.argwhere(cross_distance > r)

        if len(outside_idx) < 1:
            # no outside region cluster
            continue

        first_out, _ = find_nearest(outside_idx, row)

        dt = first_out - row
        second_idx = row - dt

        second_idx = np.clip(second_idx, 0, len(cross_distance) - 1)
        second_outside = cross_distance[second_idx] > r

        if second_outside:
            print(second_outside)
            dynamic_mask[person[:, 3] == available_times[row]] = True

        # if (cross_distance[row] > r).any():
        #     first_true_value = np.argmax(cross_distance[row] > r) # that time is dynamic
        #
        #     distant_metoids =  np.argwhere(cross_distance[row] > r)
        #
        #     for metoid_idx in distant_metoids:
        #         time_diff = available_times[metoid_idx] - available_times[row]
        #
        #         other_metoid_dynamic = False
        #
        #         # metoid comes after in timeline
        #         if time_diff > 0:
        #
        #             other_metoid_idx = available_times[row] - time_diff
        #             cross_distance[other_metoid_idx]
        #             plt.plot([available_times[row], available_times[metoid_idx], other_metoid_idx], [0,0,0], '.')
        #             plt.show()
        #
        #     if first_true_value > row:  # going forth
        #
        #
        #         time_diff = first_true_value - row  # should be times
        #         # going back, check the same time_diff to other side or the closest one if not length
        #
        #
        #         dynamic_times[first_true_value] = True
        #
        #         # distance between metoids divided by their time difference
        #         velocity_estimate = cross_distance[row][first_true_value] / time_diff # velocity per sweep (flow)
        #         # should you divide by two?
        #
        #         dynamic_mask[person[:, 3] == available_times[row]] = True
        #
        #         # if velocity_estimate < 0.1: # threshold
        #         #     velocity_estimate = -1  # do not assign because you do not know
        #
        #         if np.max(minimal_velocity_estimate[orig_mapping]) > velocity_estimate or np.max(minimal_velocity_estimate[orig_mapping]) == -1:
        #
        #             minimal_velocity_estimate[orig_mapping] = velocity_estimate
        #
        #         to_cluster_dynamic_prediction[orig_mapping] = dynamic_mask

print('processed in: ', time.time() - start_time)


# metric
GT_dyn = seq_ground_dynamic[seq_dyn == 1]    # my proposals
GT_flow_mag = np.linalg.norm(seq_flow_label[seq_dyn == 1, :3], axis=1)

# flow
GT_flow = seq_flow_label[seq_dyn == 1, :3]

# error for learning
EPE_learn = GT_flow_mag < minimal_velocity_estimate
np.mean(np.abs(GT_flow_mag[EPE_learn] - minimal_velocity_estimate[EPE_learn]))

EPE_flow = np.mean(np.abs(GT_flow_mag[minimal_velocity_estimate != -1] - minimal_velocity_estimate[minimal_velocity_estimate != -1]))

np.mean(minimal_velocity_estimate / 2 < GT_flow_mag)

v1 = visualize_points3D(seq_pts[seq_dyn==1])
v1.attributes(*[cluster_labels, to_cluster_dynamic_prediction, GT_dyn, minimal_velocity_estimate,
                np.linalg.norm(seq_flow_label[seq_dyn==1, :3],axis=1), minimal_velocity_estimate / 2 < GT_flow_mag,
                seq_pts[seq_dyn==1,3]])


visualize_points3D(to_cluster, minimal_velocity_estimate)

from sklearn.neighbors import NearestNeighbors

object_idx = 12
object_pts = to_cluster[cluster_labels == object_idx]
object_times = np.unique(object_pts[:,3])

curr_pts = object_pts[object_pts[:,3] == object_times[0]]
k_nn = NearestNeighbors(1)
k_nn.fit(curr_pts)
k_nn.n_neighbors(curr_pts)
