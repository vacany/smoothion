import numpy as np

from my_datasets.kitti.semantic_kitti import SemanticKitti_Sequence
from my_datasets.visualizer import *

seq = 27
sequence = SemanticKitti_Sequence(seq)

frame = 40
move_threshold = 0.1
# global_pts = [sequence.get_global_pts(idx, name='lidar') for idx in range(0, frame)]
# ids = [sequence.get_feature(idx, name='id_mask') for idx in range(0, frame)]
# segs = [sequence.get_feature(idx, name='lidarseg') for idx in range(0, frame)]
# dyns = [sequence.get_feature(idx, name='dynamic_label') for idx in range(0, frame)]
#
# pts = np.concatenate(global_pts)
# id_mask = np.concatenate(ids)
# seg_mask = np.concatenate(segs)
# dynamic_label = np.concatenate(dyns)

# visualize_points3D(pts, id_mask)
# visualize_points3D(pts, seg_mask)
# visualize_points3D(pts, dynamic_label)
# load all by seg,id combination

# moving_id = id_mask[dynamic_label == 1]
# moving_seg = seg_mask[dynamic_label == 1]

framerate = 10

pts1 = sequence.get_global_pts(frame, name='lidar')
pts2 = sequence.get_global_pts(frame+1, name='lidar')

dynamic_label = sequence.get_feature(frame, name='dynamic_label')

seg_mask1 = sequence.get_feature(frame, 'lidarseg')
seg_mask2 = sequence.get_feature(frame+1, 'lidarseg')

id_mask1 = sequence.get_feature(frame, 'id_mask')
id_mask2 = sequence.get_feature(frame+1, 'id_mask')

merge_id = np.concatenate((id_mask1, id_mask2))
merge_seg = np.concatenate((seg_mask1, seg_mask2))

for object_seg in np.unique(merge_seg[(merge_id > 0) & (merge_seg > 0)]):   # take only points with id and not unlabelled

    same_semantic_objects1 = pts1[seg_mask1 == object_seg]
    same_semantic_objects2 = pts2[seg_mask2 == object_seg]

    for object_id in np.unique(merge_id[merge_seg == object_seg]):  # take only pts inside the same semantic class

        # in this loop, you test every single object
        object_mask1 = (id_mask1 == object_id) & (seg_mask1 == object_seg)
        object_pts1 = pts1[object_mask1]

        object_mask2 = (id_mask2 == object_id) & (seg_mask2 == object_seg)
        object_pts2 = pts2[object_mask2]

        print(object_seg, object_id, object_pts1.shape, object_pts2.shape)

        if (np.sum(object_mask1) > 0) and (np.sum(object_mask2) > 0):

            diff = np.median(object_pts2[:,:3], axis=0) - np.median(object_pts1[:,:3], axis=0)
            velocity = np.sqrt((diff ** 2).sum() / framerate)

            if velocity > 5:
                print(f'diff ~ {diff} --- picovina')

            print('velocity: ', velocity)

            if velocity > move_threshold:
                print('Dynamic object')
            else:
                print("Static Object")


        else:
            dynamicness = -1
