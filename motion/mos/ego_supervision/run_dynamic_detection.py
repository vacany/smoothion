import os

import numpy as np

from datatools.lidar_dataset import Lidar_Dataset
from datatools.stats.segmentation import IOU

from motion_segmentation.ego_supervision.spreading import spread_ego_movement
# from ground_removal.filter_ground import pillars_ground_removal, ground_segment_velodyne


# Dataset config

DATASET_PATH = '/home/patrik/data/semantic_kitti/'

Dataset = Lidar_Dataset(dataset_type='semantic_kitti', split='train')
methods = ['ego_spreading', 'tmp_voxelization', 'rays']






tmp_metric = IOU(2)
Metric = IOU(2)

for sequence in range(11):

    max_frame = Dataset.data_info[sequence]['nbr_of_frames']
    sequence_path = Dataset.data_info[sequence]['path']
    time_range = max_frame

    for frame_id in range(0, max_frame, time_range):
        seq_min_frame, seq_max_frame = frame_id, frame_id + time_range
        if seq_max_frame > max_frame: continue

        # print(seq_min_frame, seq_max_frame)
        
        data = Dataset.get_sequence(seq_num=sequence, min_frame=seq_min_frame, max_frame=seq_max_frame, dtypes=['new_instance_label', 'seg_label', 'erasor_inference', 'ground_label', 'height_mask'])


        # Mask for support methods but from GT!
        mask = (data['ground_label'] != 1) & \
               (data['seg_label'] != 7) & \
               (data['seg_label'] != 0) & \
               (data['height_mask'] != 1)

        # mask filtered points
        points = data['points'][mask]
        dynamic_label = data['erasor_inference'][mask]
        seg_label = data['seg_label'][mask]
        # instance_label = data['instance_label'][mask]
        instance_label = data['new_instance_label'][mask]
        poses = data['sequence_poses']
        len_points = data['len_points']




        prior, infered_instace, iteration = spread_ego_movement(p=points, l=instance_label, ego_poses=poses)


        infered_by_ego = np.zeros(data['points'].shape[0])
        infered_by_ego[mask] = prior

        os.makedirs(sequence_path + '/ego_dynamic_points/', exist_ok=True)

        for t in np.unique(data['points'][:,4]):
            np.save(sequence_path + f'/ego_dynamic_points/{int(t):06d}.npy', infered_by_ego[data['points'][:,4] == t])
            # slow, keep point range? good enough so far

        print(f"Sequence {sequence} Metric ---")
        tmp_metric.update(dynamic_label, prior)
        # tmp_metric.results_to_file(sequence_path + f'/ego_dynamic_results.txt')
        tmp_metric.print_stats()
        tmp_metric.reset()


        Metric.update(dynamic_label, prior)

        # break

    print(f"All Sequences ---")
    Metric.print_stats()
