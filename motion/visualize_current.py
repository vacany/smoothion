from my_datasets import visualizer

import numpy as np
# sequence = Argoverse2_Sequence(sequence_nbr=7)

import sys

dataset = int(sys.argv[1])
seq = int(sys.argv[2])
frame = int(sys.argv[3])

if dataset == 0:
    from my_datasets.kitti.semantic_kitti import SemanticKitti_Sequence
    sequence = SemanticKitti_Sequence(sequence_nbr=seq)

elif dataset == 1:
    from my_datasets.waymo.waymo import Waymo_Sequence
    sequence = Waymo_Sequence(sequence_nbr=seq)

elif dataset == 2:
    from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence
    sequence = Argoverse2_Sequence(sequence_nbr=seq)

else:
    raise ValueError("Dataset not supported")

# local_pts = sequence.get_feature(0, name='lidar')
# v = visualizer.visualize_points3D(local_pts, local_pts[:,3])


pts1 = sequence.get_global_pts(idx=frame, name='lidar')

# time_pts = [sequence.get_global_pts(idx=idx, name='lidar') for idx in range(frame, frame+2)]
time_pts = [sequence.get_feature(idx=idx, name='lidar') for idx in range(frame, frame+2)]


# prior_pts = [sequence.get_feature(idx=idx, name='final_prior_label') for idx in range(frame, frame+50)]
# visualizer.visualize_points3D(np.concatenate(time_pts), np.concatenate(prior_pts))

freespace = []
for i in range(frame + 1, frame + 2):

    if i > 0 and i < len(sequence):
        one_freespace = sequence.get_feature(idx=i, name='accum_freespace')
        # pose = sequence.get_feature(idx=i, name='pose')
        # one_freespace = sequence.pts_to_frame(one_freespace, pose)

        freespace.append(one_freespace)
#

freespace = np.concatenate(freespace, axis=0)
# freespace = freespace[freespace[:, 2] > 0.5]

# poses = np.stack([sequence.get_feature(frame + i, name = 'pose') for i in range(-4, 5)])
# print(poses[:,:3,-1])
visualizer.visualize_multiple_pcls(*[time_pts[0][time_pts[0][:, 2] > 0.3], freespace, time_pts[1][time_pts[1][:, 2] > 0.3]])
# pts2 = np.concatenate([np.insert(sequence.get_global_pts(idx=i, name='lidar'), 4, i, 1) for i in range(0, 10)])
# visualizer.visualize_points3D(pts2, pts2[:,4])

# sys.exit('0')
# static_value = sequence.get_feature(idx=frame, name='prior_static_mask')
# dynamic_label = sequence.get_feature(idx=frame, name='dynamic_label')
# ego_prior = sequence.get_feature(idx=frame, name='corrected_ego_prior_label')
# visibility_prior = sequence.get_feature(idx=frame, name='visibility_prior')
# corrected_visibility_prior = sequence.get_feature(idx=frame, name='corrected_visibility_prior')
# correction_prior = sequence.get_feature(idx=frame, name='correction_prior')
# road_proposal = sequence.get_feature(idx=frame, name='road_proposal')
# ego_prior[road_proposal == True] = 0
# final_prior = sequence.get_feature(frame, 'final_prior_label')



# visualizer.visualize_multiple_pcls(*time_pts)
# visualizer.visualize_points3D(pts1, visibility_prior)
# visualizer.visualize_points3D(pts1, corrected_visibility_prior)
# visualizer.visualize_points3D(pts1, correction_prior)
# visualizer.visualize_points3D(pts1, correction_visibility_prior)
# visualizer.visualize_points3D(pts1, ego_prior)
# visualizer.visualize_points3D(pts1, static_value)
# visualizer.visualize_points3D(pts1, dynamic_label)
# visualizer.visualize_points3D(pts1, final_prior)
