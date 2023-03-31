import datatools.lidar_dataset
import numpy as np
import os
import glob
import open3d
import subprocess

from datatools.structures.bev import BEV
from datatools.visualizer.vis import visualize_points3D
from scipy.spatial.transform import Rotation

DATA_DIR = os.environ['HOME'] + '/data/'

def prepare_pcls_and_poses(pcls_list, poses, erasor_data_dir):
    os.makedirs(erasor_data_dir + '/pcds', exist_ok=True)

    init_static_map = []
    pose_file = open(erasor_data_dir + 'poses.csv', 'w')
    line_to_write = 'index, timestamp, x, y, z, qx, qy, qz, qw  \n'
    pose_file.writelines(line_to_write)

    for idx in range(len(pcls_list)):
        points = np.insert(pcls_list[idx], obj=3, values=1, axis=1)
        global_points = (points[:,:4] @ poses[idx].T)[:,:3]

        init_static_map.append(global_points)
        # serialize poses and correct it within the cpp main
        trans = np.round(poses[idx][:3, -1], decimals=8)
        quats = np.round(Rotation.from_matrix(poses[idx, :3, :3]).as_quat(), decimals=8)

        line_to_write = f'{idx}, {idx / 10.}, '
        line_to_write += f'{trans[0]}, {trans[1]}, {trans[2]}, {quats[0]}, {quats[1]}, {quats[2]}, {quats[3]} \n'

        pose_file.writelines(line_to_write)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points[:,:3])
        open3d.io.write_point_cloud(filename=f'{erasor_data_dir}/pcds/{idx:06d}.pcd', pointcloud=pcd, write_ascii=False)    # check name and ascci


        print(f"Frame {idx:06d} processed")

    init_static_map = np.concatenate(init_static_map)
    global_pcd = open3d.geometry.PointCloud()
    global_pcd.points = open3d.utility.Vector3dVector(init_static_map[:,:3])

    open3d.io.write_point_cloud(filename=f'{erasor_data_dir}/dense_global_map.pcd', pointcloud=global_pcd,
                                write_ascii=False)  # check name and ascci

    # pose_file.write() #?
    pose_file.close()


def compare_points_to_static_scene(pcls, points, cell_size):

    Bev = BEV(cell_size=(cell_size[0], cell_size[1]))
    Bev.create_bev_template_from_points(pcls)
    cell_z = cell_size[2]

    z_iter = np.round((pcls[:, 2].max() - pcls[:, 2].min()) / cell_z)
    z_min = pcls[:,2].min()
    inside_mask = np.zeros(points.shape[0])

    for z_idx in range(int(z_iter)):
        z_range_mask_pcls = (pcls[:, 2] > (z_min + z_idx * cell_z)) &\
                            (pcls[:, 2] < (z_min + (z_idx + 1) * cell_z))
        z_range_mask_points = (points[:, 2] > (z_min + z_idx * cell_z)) &\
                              (points[:, 2] < (z_min + (z_idx + 1) * cell_z))
        masked_points = pcls[z_range_mask_pcls]

        bev_grid = Bev.generate_bev(masked_points, features=1)

        inside_mask[z_range_mask_points] += Bev.transfer_features_to_points(points[z_range_mask_points], bev_grid)

    return inside_mask


def gather_kittiseq(sequence=4):

    sshfs_root = DATA_DIR
    sequence_path = sshfs_root + f'/semantic_kitti/dataset/sequences/{sequence:02d}/'
    rci_velodynes = sorted(glob.glob(sequence_path + 'velodyne/*.bin'))

    pcls = [np.fromfile(file_velodyne, dtype=np.float32).reshape((-1, 4)) for file_velodyne in rci_velodynes]
    poses = np.load(sequence_path + '/sequence_poses.npy')

    return pcls, poses

# def init_sshfs_on_karel():
#     subprocess.run(['sshfs - o reconnect, ServerAliveInterval = 5, nonempty, password_stdin vacekpa2@login3.rci.cvut.cz:/ rci_sshfs <<< Coldplay7'])

def main_semantic_kitti_prep():
    for sequence in range(0,11):
        print(f"Sequence --- {sequence:02d} --- Initiated!")
        pcls, poses = gather_kittiseq(sequence)
        erasor_data_dir = DATA_DIR + f'/semantic_kitti/dataset/sequences/{sequence:02d}/erasor_data_dir/data_dir/'
        prepare_pcls_and_poses(pcls, poses, erasor_data_dir=erasor_data_dir)

def main_generate_dynamic_labels(pcls, static_map, cell_size):

    static_label_list = []
    for i in range(0,len(pcls)):

        static_label = compare_points_to_static_scene(static_map, pcls[i], cell_size=cell_size)
        static_label_list.append(static_label)

        visualize_points3D(pcls[i], static_label)

        break

def run_erasor():
    print("rewrite config"
          "set up paths")

if __name__ == '__main__':
    # main_semantic_kitti_prep()
    static_map = np.asarray(
            open3d.io.read_point_cloud(
                    DATA_DIR +
                    '/semantic_kitti/dataset/sequences/04/erasor_data_dir/semantic_kitti_04_result.pcd').points)
    # visualize_points3D(static_map)
    dataset = datatools.lidar_dataset.Lidar_Dataset('semantic_kitti')

    for t in range(270,271):
        print(t)

        points = dataset.get_sequence(4, t, t)['points']
        static_points = compare_points_to_static_scene(static_map, points, cell_size=(.2,.2,.2))

        p = np.concatenate((points[static_points==True,:3],points[static_points==False,:3], static_map))
        l = np.concatenate((np.ones(points[static_points==True].shape[0]) * 2,np.ones(points[static_points==False].shape[0]), np.zeros(static_map.shape[0])))
        visualize_points3D(p, l)

    # main_generate_dynamic_labels(pcls, static_map, cell_size=(.075,.075,.075))
