import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from datatools.visualizer.vis import visualize_points3D


def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
      Args:
        pose_path: (Complete) filename for the pose file
      Returns:
        A numpy array of size nx4x4 with n poses as 4x4 transformation
        matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)


def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)

def load_labels(label_path):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half

    # sanity check
    assert ((sem_label + (inst_label << 16) == label).all())

    return sem_label, inst_label

def load_vertex(scan_path):
  """ Load 3D points of a scan. The fileformat is the .bin format used in
    the KITTI dataset.
    Args:
      scan_path: the (full) filename of the scan file
    Returns:
      A nx4 numpy array of homogeneous points (x, y, z, 1).
  """
  current_vertex = np.fromfile(scan_path, dtype=np.float32)
  current_vertex = current_vertex.reshape((-1, 4))
  current_points = current_vertex[:, 0:3]
  current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
  current_vertex[:, :-1] = current_points
  return current_vertex

# sequence = 5
#     pid = 2496839

    # dataset = Lidar_Dataset('semantic_kitti')
    # data = dataset.prep_dataset.get_data_for_erasor(sequence=sequence, start=2470, end=2500)
    # sequence_dir = DATA_DIR + f'/semantic_kitti/dataset/sequences/{sequence:02d}/'

    # erasor_proposals = [np.load(file) for file in sorted(glob.glob(sequence_dir + f'/erasor_{pid}/erasor_dynamic/*.npy'))]




sequence = 4
pcls = [load_vertex(scan_path) for scan_path in sorted(glob.glob(f'/home/patrik/data/semantic_kitti/dataset/sequences/{sequence:02d}/velodyne/*.bin'))]
results = [load_labels(scan_path)[0] for scan_path in sorted(glob.glob(f'/home/patrik/data/semantic_kitti/dataset/sequences/{sequence:02d}/mos_orig/results/*.label'))]


poses = np.load(f'/home/patrik/data/semantic_kitti/dataset/sequences/{sequence:02d}/sequence_poses.npy')

points = []
for i in range(len(poses)):
    pts = pcls[i]
    pts[:,3] = 1
    g_pts = pts @ poses[i].T

    points.append(g_pts)


points = np.concatenate(points)
results = np.concatenate(results)
visualize_points3D(points, results)
visualize_points3D(points[results > 9], results[results > 9])
from mos import MOS
pid = 0

from datatools.lidar_dataset import Lidar_Dataset
dataset = Lidar_Dataset("semantic_kitti")

data = dataset.prep_dataset.get_base_data(sequence=sequence)
pcls = data['pcls']


dyn = [np.load(file) for file in sorted(glob.glob(f'/home/patrik/data/semantic_kitti/dataset/sequences/{sequence:02d}/erasor_1/dynamic/*.npy'))]

# Mos_runner = MOS(pcls=pcls, dynamic_ids=dyn, poses=poses, output_dir=f'/home/patrik/data/semantic_kitti/dataset/sequences/{sequence:02d}/', pid=pid)
# Mos_runner.run_clustering()
# [np.save(f'/home/patrik/data/semantic_kitti/dataset/sequences/mos/mos_0/clusters/{i:06d}.npy', results[i])for i in range(len(results))]
# Mos_runner.run_tracking()
# Mos_runner.visualize_mos_tracks()

tracks = [np.load(file) for file in sorted(glob.glob(f'/home/patrik/data/semantic_kitti/dataset/sequences/{sequence:02d}/mos_1/mos_tracking/*.npy'))]
orig_tracks = [np.load(file) for file in sorted(glob.glob(f'/home/patrik/data/semantic_kitti/dataset/sequences/{sequence:02d}/mos_orig/mos_0/mos_tracking/*.npy'))]
dyn_labels = np.concatenate(data['dynamic_labels'])


from datatools.stats.segmentation import IOU
print("Our implementation of MOS")
metric = IOU(2)
metric.update(dyn_labels, np.concatenate(tracks) > 9)
metric.print_stats()

print("Mos orig Erasor data")
metric.reset()
valid_points = np.concatenate(data['valid_points'])
metric.update(dyn_labels, results[valid_points] > 0)
metric.print_stats()

print("Mos orig")
metric.reset()
valid_points = np.concatenate(data['valid_points'])
metric.update(dyn_labels, np.concatenate(orig_tracks)[valid_points] > 9)
metric.print_stats()
# tracking 0 - categories, 0 neni brana v potaz

visualize_points3D(np.concatenate(pcls), np.concatenate(orig_tracks)[valid_points])
# visualize_points3D(pcls[250], results[250])
# tracks = np.concatenate(tracks)
# visualize_points3D(points[tracks>9], tracks[tracks>9])
