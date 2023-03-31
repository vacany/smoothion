import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import open3d
import sys


DATA_DIR = os.path.expanduser("~/data/semantic_kitti/dataset/sequences/")

def load_pcl(path):
    pcd = open3d.io.read_point_cloud(path)
    pcl = np.asarray(pcd.points)
    return pcl

def merge_static_and_dynamic(path, frame):
    static = load_pcl(path + f'/scan_static/{frame:06}.bin.pcd')
    dynamic = load_pcl(path + f'/scan_dynamic/{frame:06}.bin.pcd')

    label = np.concatenate((np.zeros(static.shape[0]), np.ones(dynamic.shape[0])))

    points = np.concatenate((static, dynamic))

    both = np.concatenate((points, label[:,None]), axis=1)

    return both

def generate_sequence(sequence):
    os.makedirs(DATA_DIR + f'{sequence:02d}/removert/xyz_dyn_npy', exist_ok=True)

    files = sorted(glob.glob(DATA_DIR + f'{sequence:02d}/removert/scan_dynamic/*.pcd'))

    for i, file in enumerate(files):
        points_dyn = merge_static_and_dynamic(DATA_DIR + f'{sequence:02d}/removert/', frame=i)
        np.save(DATA_DIR + f'{sequence:02d}/removert/xyz_dyn_npy/{i:06}.npy', points_dyn)

def filter_patchworks_output(points, max_distance=0.1):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points[:, :3])
    distances, indices = nbrs.kneighbors(points[:, :3])

    return distances[:, 2] < max_distance

def patchworks_from_pcl_to_labels(data, patch_pts):

    for i in range(len(data['local_pcls'])):
        print(i)
        patch_points = patch_pts[i]
        local_points = data['local_pcls'][i]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(patch_points[:, :3])
        distances, indices = nbrs.kneighbors(local_points[:, :3])

        ground_labels = patch_points[indices[:, 0]][:, 3]

        np.save(DATA_DIR + f'/{sequence:02d}/ground_pp_label/{i:06d}', ground_labels)

def rearange_removert_for_dataloader(data):
    for i in range(len(data['local_pcls'])):
        removert_points = merge_static_and_dynamic(DATA_DIR + f'/{sequence:02d}/removert/', frame=i)
        local_points = data['local_pcls'][i]

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(removert_points[:, :3])
        distances, indices = nbrs.kneighbors(local_points[:, :3])

        frame_id = data['frames'][i]

        np.save(DATA_DIR + f'/{sequence:02d}/removert/dynamic/{frame_id:06d}', removert_points[indices[:, 0], 3])

if __name__ == "__main__":
    sequence = int(sys.argv[1])
    # generate_sequence(sequence)
    from motion_segmentation.prepare_data import Motion_Sequence
    from sklearn.neighbors import NearestNeighbors
    from datatools.visualizer.vis import visualize_points3D

    NN = NearestNeighbors(n_neighbors=1)
    os.makedirs(DATA_DIR + f'/{sequence:02d}/removert/dynamic/', exist_ok=True)

    for j in range(0, 5000, 100):   # max in semantic_kitti is below 5000 frames
        dataset = Motion_Sequence(sequence, start=j, end=j+100)
        data = dataset.get_raw_data()
        data['frames'] = list(range(j, j+100))
        print(data['frames'])
        rearange_removert_for_dataloader(data)
