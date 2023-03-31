import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from shutil import copytree, rmtree
import hdbscan


from tracking.kalman_tracking import run_multi_object
from datatools.structures.point import get_farthest_points, get_max_size
from datatools.lidar_dataset import Lidar_Dataset
from datatools.visualizer.vis import visualize_points3D

DATA_DIR  = os.environ['HOME'] + '/data/'



def mos_config():
    config = {
              'cluster_epsilon' : 0.5,
              'max_object_size' : 20,
             }

    return config

# create tmp directory
class MOS():
    def __init__(self, pcls, dynamic_ids, poses, output_dir, pid=None, **kwargs):
        self.pcls = pcls
        self.dynamic_ids = dynamic_ids
        self.poses = poses

        self.output_dir = output_dir
        self.cfg = mos_config()
        for k, v in kwargs.items():
            self.cfg[k] = v

        self.pid = pid if pid is not None else os.getpid()

        self.output_dir = self.output_dir + f'/mos_{self.pid}/'

        self.output_dir_tracking = self.output_dir + '/mos_tracking/'
        self.output_dir_clusters = self.output_dir + '/clusters/'
        self.output_dir_dynamic = self.output_dir + '/automos_dynamic/'

        for folder in [self.output_dir,
                       self.output_dir_tracking,
                       self.output_dir_clusters,
                       self.output_dir_dynamic]:

            os.makedirs(folder, exist_ok=True)



    def run_clustering(self):

        ids_list = []

        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_epsilon=2, max_cluster_size=10)#cluster_selection_epsilon=self.cfg['cluster_epsilon'])

        for frame_id in range(0, len(self.pcls)):
            # if os.path.exists(self.output_dir_clusters + f'{frame_id:06d}.npy'):
            #     print(self.output_dir_clusters + f'{frame_id:06d}.npy', ' exists ---> skipping!')
            #     continue

            pcl = self.pcls[frame_id]
            ids = np.zeros(pcl.shape[0], dtype=int) # zero is not taken in tracking?

            dynamic_inference = self.dynamic_ids[frame_id]
            # skip if there is nothing to cluster
            print(frame_id, np.sum(dynamic_inference))
            if np.sum(dynamic_inference) < 3:
                np.save(self.output_dir_clusters + f'{frame_id:06d}.npy', ids)
                continue

            clusterer.fit(pcl[dynamic_inference == 1, :3])
            ids[dynamic_inference == 1] = clusterer.labels_ + 1
            ids[ids < 1] = 0


            for id in np.unique(ids):
                size = get_max_size(pcl[ids == id, :2])

                if size > self.cfg['max_object_size']:
                    # print("Max size exceeded")
                    print(id)
                    # ids[ids == id] = 0

                if len(ids[ids == id]) < 5:
                    print('wut', id, len(ids[ids == id]))


            ids_list.append(ids)
            np.save(self.output_dir_clusters + f'{frame_id:06d}.npy', ids)

    def run_tracking(self):
        # ids are out of function arguments, loading from previouse clustering!
        ids = [np.load(file) for file in sorted(glob.glob(self.output_dir_clusters + f'*.npy'))]

        run_multi_object(output_dir=self.output_dir_tracking,
                         pcls=self.pcls,
                         ids=ids,
                         poses=self.poses
                                         )


    def visualize_mos_tracks(self):
        ids = [np.load(file) for file in sorted(glob.glob(self.output_dir_clusters + f'*.npy'))]
        pcls = np.concatenate(data['pcls'])
        tracks = np.concatenate([np.load(file) for file in sorted(glob.glob(self.output_dir_tracking + '/*.npy'))])

        # visualize_points3D(pcls, tracks)
        visualize_points3D(pcls, np.concatenate(ids))
        visualize_points3D(pcls, tracks)


if "__main__" == __name__:
    # sequence = 4
    # pid = 2497007
    import sys

    sequence = int(sys.argv[1])
    print(sequence)
    # sequence = 4
    # pid = 2496839
    pid = 1
    dataset = Lidar_Dataset('semantic_kitti')
    data = dataset.prep_dataset.get_base_data(sequence=sequence)
    sequence_dir = DATA_DIR + f'/semantic_kitti/dataset/sequences/{sequence:02d}/'
    print(sequence)

    erasor_proposals = [np.load(file) for file in sorted(glob.glob(sequence_dir + f'/erasor_{pid}/dynamic/*.npy'))]

    Mos_runner = MOS(pcls=data['local_pcls'], dynamic_ids=erasor_proposals, poses=data['poses'], output_dir=sequence_dir, pid=pid, erasor_dir=f'erasor_{pid}')
    Mos_runner.run_clustering()
    Mos_runner.run_tracking()
    # Mos_runner.visualize_mos_tracks()




# TODO MOS solve Tracking
# TODO DO method where ego spread labels with worm hole, remove lowest points (road), cluster surrounding and track the right object


# from datatools.stats.segmentation import IOU
# metric = IOU(2)

# generated_labels = np.concatenate([np.load(file) for file in sorted(glob.glob(Mos_runner.output_dir_tracking + '/*.npy'))])


# mos_inference = dyn_list * dyn_labels_list
# ego = ego_dynamic_points_list
# eras_ego = (dyn_list + ego_dynamic_points_list) >= 1
# eras_mos_ego = (mos_inference + dyn_list + ego_dynamic_points_list) >= 1
# eras_ego_mos_ego = (eras_ego + mos_inference) >= 1
# metric.update(dyn_labels_list, ego)
# metric.print_stats()

# metric, connect ego dynamic points or estimate?, tracking, scheme for summary, table
# One set of hyperparams,
# run_multi_object_kalman_tracking(output_dir_tracking, pcls, ids_files, poses)
