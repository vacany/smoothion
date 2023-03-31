import os

from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA

from datatools.structures.geometry import points_in_hull
from datatools.visualizer.vis import *
from datatools.structures.bev import BEV
from datatools.structures.box import get_ego_bbox, get_point_mask, get_boxes_from_ego_poses

from motion_segmentation.prepare_data import Motion_Sequence
from motion_segmentation.ego_supervision.common import split_concatenation



def get_ego_points(poses, cell_size=(0.1, 0.1)):

    coors = []
    box = get_ego_bbox()
    for pose in poses:

        y = np.linspace(-box[4] / 2, box[4] / 2, int(box[4] / cell_size[1] * 2))

        for j in y:
            x = np.linspace(- box[3] / 2, box[3] / 2, int(box[3] / cell_size[0] * 2))
            ego_points = np.insert(x[:, np.newaxis], obj=1, values=j, axis=1)
            ego_points = np.insert(ego_points, obj=2, values=1, axis=1)
            # Rotation
            ego_points = ego_points @ pose[:3, :3].T
            # Translation
            ego_points[:, :2] = ego_points[:, :2] + pose[:2, -1]

            coors.append(ego_points)

    coors = np.concatenate(coors)

    return coors

def fit_ground_plane_under_ego(points, pose=None, min_pts=3, tau_margin=0.25, iterations=3):
    if len(points) < min_pts:
        return np.zeros(points.shape[0])

    pca_model = PCA(n_components=3)
    z_mean = points[:, 2].mean()
    # init mask
    tau = tau_margin
    ground_bin_mask = points[:, 2] < z_mean + tau #-1.73 - pose[0,2,-1] * 0.9 # TODO figure this out

    for _ in range(iterations):
        if np.sum(ground_bin_mask) <= 2:
            ground_bin_mask = np.zeros(points.shape[0], dtype=bool)
            break

        pca_model.fit(points[ground_bin_mask, :3])
        col_idx = np.argmin(pca_model.explained_variance_)

        n_vector = pca_model.components_[:, col_idx]
        d_mean = - n_vector.T @ points[ground_bin_mask, :3].mean(0)
        d_dash = - n_vector.T @ points[ground_bin_mask, :3].T



        ground_bin_mask[ground_bin_mask] = d_mean - d_dash < tau_margin

    # visualize_plane_with_points(points[ground_bin_mask], n_vector, d_mean)
    ground_bin_mask = ground_bin_mask

    # TODO this adds noise on edges?
    # Bev = BEV((0.2, 0.2))
    # Bev.create_bev_template_from_points(points)
    # height_grid = Bev.generate_bev(points[ground_bin_mask], points[ground_bin_mask][:, 2])
    # new_height = Bev.transfer_features_to_points(points, height_grid)
    # ground_bin_mask[new_height >= points[:, 2]] = True

    return ground_bin_mask

def ego_safest_run(points_M, poses):

    ego_boxes = get_boxes_from_ego_poses(poses)
    dynamic_inference = - np.ones(points_M.shape[0], dtype=int)

    for idx in range(len(poses) - 1):
        print(idx)

        intersected_points = get_point_mask(points_M, ego_boxes[idx])
        dynamic_inference[intersected_points] = 1

    return dynamic_inference


def ego_odometry_run(points_M, poses):
    '''

    :param points_M:  Global map
    :param poses: poses N x 4x4
    :return: Mask of dynamic, ground and unclassified points [1,0,-1]
    '''
    dynamic_inference = - np.ones(points_M.shape[0], dtype=int)
    for idx in range(len(poses) - 1):
        print(idx)
        ego_pts = get_ego_points(poses[idx:idx + 1])
        intersected_points = points_in_hull(points_M, ego_pts)

        ground = fit_ground_plane_under_ego(points_M[intersected_points], poses[idx:idx + 1])

        dynamic_inference[intersected_points] = (ground==False)

    return dynamic_inference

def area_around_pose(pts, poses, radius):
    Bev = BEV(cell_size=(0.1,0.1))
    Bev.create_bev_template_from_points(pts)

    x = np.linspace(-radius, radius, int(2 * radius / 0.1) + 10)
    xx,yy = np.meshgrid(x, x)
    xx = np.concatenate(xx)
    yy = np.concatenate(yy)

    circle_mask = xx ** 2 + yy ** 2 < radius ** 2
    raster_points = np.concatenate((xx[:,None], yy[:,None], np.ones(xx[:,None].shape)), axis=1)

    grid = np.zeros(Bev.shape, dtype=bool)
    for pose in poses:
        tmp_grid = Bev.generate_bev(raster_points[circle_mask] + pose[:3,-1], True)
        grid = grid + tmp_grid

    grid = grid > 0

    points_around_poses = Bev.transfer_features_to_points(pts, grid)

    return points_around_poses


def filter_patchworks_output(points, max_distance=0.1):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points[:, :3])
    distances, indices = nbrs.kneighbors(points[:, :3])

    return distances[:, 2] < max_distance



def cluster_main_road(points, eps=0.1, min_samples=5):
    clu = DBSCAN(eps=eps, min_samples=min_samples)
    clu.fit(points[:, :3])
    ground_clusters = clu.labels_
    # road is the biggest
    u,c = np.unique(ground_clusters, return_counts=True)
    largest_idx = np.argmax(c)
    road_cluster_idx = u[largest_idx]

    return ground_clusters == road_cluster_idx



def split_sequence_inference(sequence_path):

    os.makedirs(sequence_path + '/ego_odometry', exist_ok=True)
    os.makedirs(sequence_path + '/ego_safest', exist_ok=True)

    ego_odometry = np.load(sequence_path + f"/ego_odometry.npy")
    ego_safest = np.load(sequence_path + f"/ego_safest.npy")

    for frame in range(len(data['pcls'])):
        print(frame)
        if frame == 0:
            mask = (0, data['pcls_point_range'][frame])
        else:
            mask = (data['pcls_point_range'][frame - 1], data['pcls_point_range'][frame])

        ego_odo = ego_odometry[mask[0] : mask[1]]
        ego_safe = ego_safe[mask[0] : mask[1]]

        assert data['pcls'][frame].shape != ego_odo.shape
        assert data['pcls'][frame].shape != ego_safest.shape

        np.save(sequence_path + f'/ego_odometry/{frame:06d}.npy', ego_odo)
        np.save(sequence_path + f'/ego_safest/{frame:06d}.npy', ego_safest)

def separate_ground(pts, patchworks_output, poses, radius):
    '''

    :param pts: Whole pcls
    :param poses: corresponding poses, maybe all poses as well?
    :param radius: considered Radius of area aroung ego
    :return:
    '''

    points_around_poses = area_around_pose(pts, poses, radius=radius)

    coarse_ground_mask = points_around_poses & patchworks_output
    coarse_ground_idx = np.argwhere(coarse_ground_mask)[:, 0]

    ###FIlter, might be part of ablation
    # biggest is the road
    ground_cluster = cluster_main_road(points=pts[coarse_ground_idx], eps=EPS)

    # visualize_points3D(pts[coarse_ground_idx], ground_cluster)

    ground_mask = np.zeros(pts.shape[0], dtype=bool)
    ground_mask[coarse_ground_idx] = ground_cluster

    ground_noise = np.zeros(pts.shape[0], dtype=bool)
    ground_noise[coarse_ground_idx] = ground_cluster == False

    return ground_mask, ground_noise

def select_points_above_ground_area(pts, ground_mask, ground_noise):
    '''

    :param pts: Whole pcls
    :param ground_mask: fine ground mask
    :param ground_noise: eliminated noise from ground mask
    :return:
    '''
    non_ground_idx = np.argwhere(ground_mask == False)[:, 0]
    non_noise_idx = np.argwhere(ground_noise == False)[:, 0]

    u, c = np.unique(np.concatenate((non_ground_idx, non_noise_idx)), return_counts=True)
    valid_idx = u[c == 2]

    # project from 2D
    Bev = BEV(cell_size=(0.02, 0.02))
    Bev.create_bev_template_from_points(pts)
    interest_area_grid = Bev.generate_bev(pts[ground_mask], features=True)
    potentially_dynamic = Bev.transfer_features_to_points(pts[valid_idx], interest_area_grid)
    without_ground = np.array(potentially_dynamic * (ground_mask[valid_idx] == False), dtype=bool)

    above_ground = np.zeros(pts.shape[0], dtype=bool)
    above_ground[valid_idx] = without_ground

    return above_ground


if __name__ == '__main__':


    import sys
    # odometry spreding <----
    DATASET_TYPE = 'semantic_kitti'
    sequence = int(sys.argv[1])
    DATA_DIR = os.environ['HOME'] + f'/data/semantic_kitti/dataset/sequences/{sequence:02d}/'
    start = 0
    end = np.inf

    # Hyperparms
    # patchworks default
    EPS = 0.1
    RADIUS_range = [0.5, 1, 1.5, 2., 2.5, 3.5, 4.]

    for RADIUS in RADIUS_range:
        print("Sequence: ", sequence)

        dataset = Motion_Sequence(sequence, start=start, end=end)

        ###Preload Data
        data = dataset.get_raw_data()

        pcls = data['pcls']
        poses = data['poses']
        valid_mask = np.concatenate(data['valid_points'])

        # ego_safest = dataset.get_specific_data('/ego_safest/*.npy', form='concat')
        # ego_odometry = dataset.get_specific_data('/ego_odometry/*.npy', form='concat')

        ###Ground Truth
        dynamic_labels = np.concatenate(data['dynamic_labels'])
        potential_mask = np.zeros(dynamic_labels.shape[0], dtype=bool)
        ###PATCHWORKS
        pcls = [np.insert(pcls[i], 3, i, axis=1) for i in range(len(pcls))]
        # prepare data
        pts = np.concatenate(pcls)

        patchworks = dataset.get_specific_data('/ground_label/*.npy', form='concat')[valid_mask]
        ###Only around Ego

        ground_mask, ground_noise = separate_ground(pts, patchworks_output=patchworks, poses=poses, radius=RADIUS)

        dynamic_proposals = select_points_above_ground_area(pts, ground_mask, ground_noise)

        dynamic_list = split_concatenation(data, dynamic_proposals)

        os.makedirs(f'{DATA_DIR}/ego_radius/', exist_ok=True)
        os.makedirs(f'{DATA_DIR}/ego_radius/radius_{RADIUS}', exist_ok=True)

        for i, dynamic in enumerate(dynamic_list):
            np.save(f'{DATA_DIR}/ego_radius/radius_{RADIUS}/{i:06d}.npy', dynamic)
        np.save(f'{DATA_DIR}/ego_radius/radius_{RADIUS}_proposals.npy', dynamic_proposals)

    # visualize_points3D(pts, dynamic_proposals)


    # TODO Ego spread
    # 1) remove ground (pp with validation clustering) \tick
    # 2) cluster xyt, find parameter (grid search?) \tick
    # 3) When ego interascts, just propage the clusters
    # 4) show when above PP without ego intersection
    # 5) try tracking

    # Connect Ego safest, Ego small radius, Ego Spread

    # ###Clustering
    # import hdbscan
    # clusterer = hdbscan.HDBSCAN()
    #
    # cluster_mask = - np.ones(pts.shape[0])
    # for t in np.unique(pts[:,3]):
    #
    #     print(t)
    #     tmp_mask = (pts[:, 3] == t) & dynamic_proposals
    #     if np.sum(tmp_mask) < 3: continue
    #     clusterer.fit(pts[tmp_mask])
    #     clusters = clusterer.labels_
    #     cluster_mask[tmp_mask] = clusters

    # visualize_points3D(pts[dynamic_proposals], cluster_mask[dynamic_proposals])   # per frame --- REFAK

    ###Tracking
    # from tracking.kalman_tracking import run_multi_object
    #
    # # Prepare data
    # # def track
    # local_pcls = data['local_pcls']
    # ids = []
    # start_idx = 0
    # for i in range(len(local_pcls)):
    #     current_cluster_mask = cluster_mask[start_idx : data['pcls_point_range'][i]]
    #     ids.append(current_cluster_mask)
    #     start_idx += len(current_cluster_mask)
    #
    # tracking_data = run_multi_object('track', pcls=local_pcls, poses=poses, ids=ids)
    # tracked_inference = tracking_data['mos_labels']
    # tracks_pts = np.concatenate(tracked_inference)
    #
    # visualize_points3D(pts, tracks_pts > 9)
    # visualize_points3D(pts, without_ground)
    # metric.update(dynamic_labels, tracks_pts > 9)
    # metric.print_stats()
