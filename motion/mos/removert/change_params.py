import sys
import yaml
import os
import glob

with open('orig_params_kitti.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)


dataset_type = 'semantic_kitti'
pat_rci_path = f'/home/patrik/mnt/rci/mnt/personal/vacekpa2/data/{dataset_type}/dataset/sequences/'


for sequence in range(12,13):
    with open('orig_params_kitti.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    length_of_seq = len(os.listdir('/home/patrik/mnt/rci/mnt/personal/vacekpa2/data/semantic_kitti/dataset/sequences/' + f'{sequence:02d}/velodyne/'))

    os.makedirs(f'{pat_rci_path}/{sequence:02d}/removert/scan_dynamic/', exist_ok=True)

    # files_processed = len(glob.glob(f'{pat_rci_path}/{sequence:02d}/removert/scan_dynamic/*'))

    rem = config['removert']

    rci_path = '/local/vacekpa2/rci_sshfs/mnt/personal/vacekpa2/data/semantic_kitti/dataset/sequences/' # make sure sshfs is up
    rem['save_pcd_directory'] = rci_path + f"{sequence:02d}/removert/"
    rem['sequence_scan_dir'] = rci_path + f"{sequence:02d}/velodyne/"
    rem['sequence_pose_path'] = rci_path + f"{sequence:02d}/poses.txt"

    rem['sequence_vfov'] = 50
    rem['sequence_hfov'] = 360

    rem['use_keyframe_gap'] = False
    rem['keyframe_gap'] = 1
    rem['batch_size'] = 50  # keep?

    for i in range(0, length_of_seq, 50):
        rem['start_idx'] = i
        rem['end_idx'] = i + 50

        # karel_path = '/home/patrik/mnt/karel/local/vacekpa2/catkin_ws/removert_ws/src/removert/config/'
        karel_path = '/home/patrik/mnt/karel/local/vacekpa2/catkin_ws/removert_ws/pat/configs/'

        with open(karel_path + f'/{sequence:02d}_{i:06d}_params_kitti.yaml', 'w') as g:
            yaml.dump({'removert' : rem}, g)

        print(rem)
