import os

import hdbscan
from datatools.structures.box import get_ego_points
from motion_segmentation.ego_supervision.common import spatio_temporal_clustering, split_concatenation

from datatools.structures.geometry import points_in_hull
from datatools.visualizer.vis import *
from datatools.stats.segmentation import IOU

from motion_segmentation.prepare_data import Motion_Sequence


def spread_ego_movement(p, l, ego_poses, MIN_POINTS_PER_BLUR=30):
    '''

    :param p: whole sequence point cloud [N] x [x,y,time]
    :param l: clustered id mask
    :param ego_poses: ego_poses
    :return: prior_mask for dynamic points
    '''
    Ego_points = get_ego_points(ego_poses)

    remaining_point_mask = np.ones(l.shape, dtype=bool)
    prior_point_mask = np.zeros(l.shape, dtype=bool)

    # List of objects
    list_of_objects = [Ego_points]
    iteration_idx = np.zeros(p.shape[0])
    instance_idx = np.zeros(p.shape[0])

    instance_count = 1
    iteration = 1

    # visualize_points3D(np.concatenate((Ego_points[:,:3], p[:,:3])), np.concatenate((np.ones(Ego_points.shape[0]), np.zeros(p.shape[0]))))

    # Main loop
    while len(list_of_objects) > 0:

        dynamic_points = list_of_objects.pop(0)

        # Spreding clusters
        hull_mask = np.zeros(l.shape, dtype=bool)

        try:
            hull_mask[remaining_point_mask] = points_in_hull(p[remaining_point_mask, :2], dynamic_points[:, :2])
        except:
            print(f"Cannot create convex hull from these points - check in future")
            continue

        connected_clu_ids = np.unique(l[hull_mask])
        connected_clu_ids = connected_clu_ids[connected_clu_ids != -1]  # eliminate noise

        # Feeding next objects
        for clu_id in connected_clu_ids:
            # Getting object points
            object_mask = l == clu_id
            object_points = p[object_mask]

            # updating memory masks
            remaining_point_mask[object_mask] = False
            prior_point_mask[object_mask] = True
            iteration_idx[object_mask] = iteration
            instance_idx[object_mask] = instance_count

            # Fill values for dynamic class
            object_points[:,2] = 0  # z-axis
            object_points = np.insert(object_points, obj=3, values=p[object_mask, 2], axis=1)

            # Check for reasonable size of object
            if len(object_points) < MIN_POINTS_PER_BLUR:
                continue

            # Update list for next iterations
            list_of_objects.append(object_points)

            instance_count += 1

        print(f'Generation {iteration} \t length of list: {len(list_of_objects)}')
        iteration += 1

    return prior_point_mask, instance_idx

def spread_argoverse_sequence(ArgoSeq):
    '''

    :param ArgoSeq:
    :param visualize:
    :return:
    '''
    poses = ArgoSeq.get_all_poses()

    pcl, clusters = ArgoSeq.get_pcl_with_clusters()
    hd_mask = ArgoSeq.mask_ground_and_height(pcl)
    gt = ArgoSeq.get_labels(0, len(ArgoSeq))[hd_mask, 2]
    pcl = pcl[hd_mask]
    clusters = clusters[hd_mask]

    prior, instance_idx, iteration_idx = spread_ego_movement(pcl, clusters, poses)

    return pcl, gt, prior, instance_idx, iteration_idx

def unpack_ArgoSeq(ArgoSeq, start=0, end=None):
    ''' dev helper '''
    if end is None:
        end = len(ArgoSeq)

    poses = ArgoSeq.get_all_poses()

    pcl, clusters = ArgoSeq.get_pcl_with_clusters()
    hd_mask = ArgoSeq.mask_ground_and_height(pcl)
    hd_mask = (pcl[:,4] >= start) & (pcl[:,4] <= end) & hd_mask

    gt = ArgoSeq.get_labels(0, len(ArgoSeq))[hd_mask]
    pcl = pcl[hd_mask]
    clusters = clusters[hd_mask]
    return pcl, clusters, gt, poses


if __name__ == '__main__':

    from sklearn.model_selection import ParameterGrid

    # odometry spreding <----
    DATASET_TYPE = 'semantic_kitti'
    sequence = int(sys.argv[1])
    DATA_DIR = os.environ['HOME'] + f'/data/semantic_kitti/dataset/sequences/{sequence:02d}/'

    max_frames = len(os.listdir(DATA_DIR + '/velodyne'))

    param_grid = {'START' : [i for i in range(0, max_frames, 4)],
                  'EPS': [0.5, 0.8, 1., 1.5, 2., 2.5],
                  'TIME_REACH': [2, 3],
                  }

    grid = ParameterGrid(param_grid)
    print("Sequence: ", sequence)

    for exp_nbr, params in enumerate(grid):
        print(exp_nbr)
        print(params['EPS'], params['TIME_REACH'], params['START'])

        ###Hyperparms
        # patchworks default
        EPS = params['EPS']
        TIME_REACH = params['TIME_REACH']
        start = params["START"]
        end = start + 4

        exp_path = DATA_DIR + f'/ego_spreading/EPS_{EPS:.2f}_TIME_REACH_{TIME_REACH}'
        os.makedirs(exp_path, exist_ok=True)

        if os.path.exists(exp_path + f'/START_{start}_END_{end}.npy'):
            continue

        dataset = Motion_Sequence(sequence, start=start, end=end)

        ###Preload Data
        data = dataset.get_raw_data()

        pcls = data['pcls']
        poses = data['poses']
        valid_mask = np.concatenate(data['valid_points'])

        ###Ground Truth
        dynamic_labels = np.concatenate(data['dynamic_labels'])
        potential_mask = np.zeros(dynamic_labels.shape[0], dtype=bool)

        ###PATCHWORKS
        pcls = [np.insert(pcls[i], 3, i, axis=1) for i in range(len(pcls))]
        # prepare data
        pts = np.concatenate(pcls)

        patchworks = dataset.get_specific_data('/ground_label/*.npy', form='concat')[valid_mask]
        cluster_mask = - np.ones(pts.shape[0], dtype=int)

        non_ground_dx = np.argwhere(patchworks==False)[:,0]

        try:
            pts_no_ground = np.insert(pts[non_ground_dx], obj=3, values=pts[non_ground_dx,3], axis=1)
            clusters = spatio_temporal_clustering(pts_no_ground, eps=EPS, time_reach=TIME_REACH)
            cluster_mask[non_ground_dx] = clusters
        except:
            print(params, 'probably oom error')
            continue

        # store clusters,

        np.save(exp_path + f'/START_{start}_END_{end}.npy', cluster_mask)

        ids = split_concatenation(data, cluster_mask)
        for folder in ['clusters', 'priors','instances']:
            os.makedirs(exp_path + f'/{folder}/', exist_ok=True)

        ###SPREADING
        # prior, instance = spread_ego_movement(pts_no_ground, cluster_mask[non_ground_dx], poses, MIN_POINTS_PER_BLUR=10)

        # prior_list = split_concatenation(data, prior)
        # instance_list = split_concatenation(data, instance)

        for i, frame_id in enumerate(range(start, end)):
            if frame_id >= max_frames: continue

            np.save(exp_path + f'/clusters/{frame_id:06d}.npy',ids[i])
            # np.save(exp_path + f'/priors/{frame_id:06d}.npy',prior_list[i])
            # np.save(exp_path + f'/instances/{frame_id:06d}.npy',instance_list[i])
