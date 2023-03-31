import sys

import hdbscan
import numpy
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from datatools.structures.bev import BEV

def split_concatenation(data, cluster_mask):

    ids = []
    start_idx = 0
    for i in range(len(data['pcls_point_range'])):
        current_cluster_mask = cluster_mask[start_idx : data['pcls_point_range'][i]]
        ids.append(current_cluster_mask)
        start_idx += len(current_cluster_mask)

    return ids


def _old_spatio_temporal_clustering(pcls, eps=0.3, time_reach=3, method="DBSCAN"):
    tmp_pcls = pcls.copy()
    ''' Time scalling '''
    time_scale = eps / time_reach - 0.01
    tmp_pcls[:,4] *= time_scale
    if method.casefold() == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        clusters = clusterer.fit_predict(tmp_pcls[:,[0,1,4]])   # x,y,time

    if method.casefold() == 'dbscan':
        clusterer = DBSCAN(eps=eps, min_samples=3, algorithm='ball_tree')
        clusters = clusterer.fit_predict(tmp_pcls[:,[0,1,4]])

    return clusters

def patchworks_from_pcl_to_labels(data, patch_pts):

    for i in range(len(data['local_pcls'])):
        print(i)
        patch_points = patch_pts[i]
        local_points = data['local_pcls'][i]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(patch_points[:, :3])
        distances, indices = nbrs.kneighbors(local_points[:, :3])

        ground_labels = patch_points[indices[:, 0]][:, 3]

        np.save(DATA_DIR + f'/{sequence:02d}/ground_pp_label/{i:06d}', ground_labels)

def mask_by_discretized_nn(pts, points, cell_size):
    '''

    :param pts: Points to be masked
    :param points: Anchor points that are masking surroundings
    :param cell_size:
    :return:
    '''
    Bev = BEV(cell_size=(cell_size[0], cell_size[1]))
    Bev.create_bev_template_from_points(pts)
    cell_z = cell_size[2]

    z_iter = np.round((pts[:, 2].max() - pts[:, 2].min()) / cell_z)
    z_min = pts[:,2].min()
    inside_mask = np.zeros(pts.shape[0])

    for z_idx in range(int(z_iter)):
        print(z_idx + 1, '/', int(z_iter))
        z_range_mask_pts = (pts[:, 2] > (z_min + z_idx * cell_z)) &\
                            (pts[:, 2] < (z_min + (z_idx + 1) * cell_z))
        z_range_mask_points = (points[:, 2] > (z_min + z_idx * cell_z)) &\
                              (points[:, 2] < (z_min + (z_idx + 1) * cell_z))

        masked_points = points[z_range_mask_points]

        bev_grid = Bev.generate_bev(masked_points, features=1)

        inside_mask[z_range_mask_pts] += Bev.transfer_features_to_points(pts[z_range_mask_pts], bev_grid)


    return np.array(inside_mask, dtype=bool)

#TODO Init clusters? for running on longer sequences
def spatio_temporal_clustering(data, ground_list, clusterer):
    '''

    :param data: data from dataloader
    :param ground_list: list of patchworks output
    :param clusterer: model - DBSCAN/HDBSCAN
    :return:
    '''

    cluster_list = []

    first_pts = data['pcls'][0]
    second_pts = data['pcls'][1]

    first_non_ground_idx = np.argwhere(ground_list[0] == False)[:, 0]
    second_non_ground_idx = np.argwhere(ground_list[1] == False)[:, 0]

    two_pts = np.concatenate((first_pts[first_non_ground_idx], second_pts[second_non_ground_idx]))

    clusters = clusterer.fit_predict(two_pts[:, :3])
    indices_for_next = clusters[first_non_ground_idx.shape[0]:].copy()

    first_cluster_mask = - np.ones(first_pts.shape[0], dtype=int)
    second_cluster_mask = - np.ones(second_pts.shape[0], dtype=int)

    first_cluster_mask[first_non_ground_idx] = clusters[:first_non_ground_idx.shape[0]].copy()
    second_cluster_mask[second_non_ground_idx] = clusters[first_non_ground_idx.shape[0]:].copy()

    cluster_list.append(first_cluster_mask)


    ### Loop rest
    for i in range(1, len(data['pcls']) - 1):
        print(i)
        final_cluster_mask = - np.ones(data['pcls'][i].shape[0], dtype=int)

        first_pts = data['pcls'][i]
        second_pts = data['pcls'][i + 1]

        first_non_ground_idx = np.argwhere(ground_list[i] == False)[:, 0]
        second_non_ground_idx = np.argwhere(ground_list[i + 1] == False)[:, 0]

        two_pts = np.concatenate((first_pts[first_non_ground_idx], second_pts[second_non_ground_idx]))
        clusters = clusterer.fit_predict(two_pts[:, :3])

        tmp_max_idx = np.max(indices_for_next)  # todo!
        memory_mask = np.zeros(clusters.shape, dtype=bool)
        new_cluster_mask = - np.ones(clusters.shape, dtype=int)

        corresponding_clusters = clusters[:first_non_ground_idx.shape[0]].copy()
        # Assing first pointcloud previous ids
        for propagate_id in np.unique(indices_for_next):
            if propagate_id == -1: continue

            previous_mask = indices_for_next == propagate_id

            found_indices = corresponding_clusters[previous_mask].copy()

            for corr_id in np.unique(found_indices):
                if corr_id == -1: continue
                new_cluster_mask[clusters == corr_id] = propagate_id
                memory_mask[clusters == corr_id] = True


        indices_for_next = new_cluster_mask[first_non_ground_idx.shape[0]:]
        final_cluster_mask[first_non_ground_idx] = new_cluster_mask[:first_non_ground_idx.shape[0]]

        cluster_list.append(final_cluster_mask)

    return cluster_list

if __name__ == '__main__':
    from motion_segmentation.prepare_data import Motion_Sequence
    from datatools.visualizer.vis import visualize_points3D
    sequence = int(sys.argv[1])
    dataset = Motion_Sequence(sequence=sequence, start=240)
    data = dataset.get_raw_data()
    ground_list = dataset.get_specific_data('ground_pp_label/*.npy')
    EPS = 0.5
    clusterer = DBSCAN(eps=EPS, min_samples=10)

    # script to run patchworks and clustering different eps
    pts = np.concatenate(data['pcls'])
    ground_M = np.concatenate(ground_list)
    inside = mask_by_discretized_nn(pts, pts[ground_M == True], cell_size=(0.5, 0.5, 0.3))

    # per cluster ground removal and validation cluster? ...
    new_ground_list = []
    for i in range(len(ground_list)):
        if i == 0:
            new_ground_list.append(inside[:data['pcls_point_range'][0]])
        else:
            first = data['pcls_point_range'][i-1]
            next = data['pcls_point_range'][i]
            new_ground_list.append(inside[first : next])


    cluster_list = spatio_temporal_clustering(data, new_ground_list, clusterer=clusterer)

    seq_path = dataset.dataset.prep_dataset.data_info[sequence]['path']
    os.makedirs(seq_path + '/spat_temp_cluster/', exist_ok=True)

    for i in range(len(cluster_list)):
        np.save(seq_path + f'/spat_temp_cluster/{i:06d}.npy', cluster_list[i])
