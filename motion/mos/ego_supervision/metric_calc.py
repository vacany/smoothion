import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from datatools.stats.segmentation import IOU
from datatools.visualizer.vis import visualize_points3D

from motion_segmentation.prepare_data import Motion_Sequence

if __name__ == '__main__':

    DATASET_TYPE = 'semantic_kitti'
    sequence = 1
    DATA_DIR = os.environ['HOME'] + f'/data/semantic_kitti/dataset/sequences/{sequence:02d}/'
    start = 0
    end = 100
    metric = IOU(2)
    pid = 1
    print("Sequence: ", sequence)


    dataset = Motion_Sequence(sequence, start=start, end=end)

    data = dataset.get_raw_data()
    poses = data['poses']
    # ego_safest = dataset.get_specific_data('/ego_safest/*.npy', form='concat')
    # ego_odometry = dataset.get_specific_data('/ego_odometry/*.npy', form='concat')
    clusters = dataset.get_specific_data('/spat_temp_cluster/*.npy', form='concat')
    removert = dataset.get_specific_data('/removert/dynamic/*.npy', form='concat')

    #Ground Truth
    dynamic_labels = np.concatenate(data['dynamic_labels'])
    pcls = np.concatenate(data['pcls'])




    visualize_points3D(pcls, removert)
    for name, inference in zip(['removert'], [removert]):
        print(f"{name} ----> ")
        metric.reset()
        metric.update(dynamic_labels, inference==1)
        metric.print_stats()


###Tracking
    from tracking.kalman_tracking import run_multi_object

    # Prepare data
    # def track
    local_pcls = data['local_pcls']
    ids = []
    start_idx = 0
    for i in range(len(local_pcls)):
        current_cluster_mask = clusters[start_idx : data['pcls_point_range'][i]]
        ids.append(current_cluster_mask)
        start_idx += len(current_cluster_mask)

    tracking_data = run_multi_object('track', pcls=local_pcls, poses=poses, ids=ids)
    tracked_inference = tracking_data['mos_labels']
    tracks_pts = np.concatenate(tracked_inference)

    visualize_points3D(pcls, tracks_pts > 9)
    metric.update(dynamic_labels, tracks_pts > 9)
    metric.print_stats()
