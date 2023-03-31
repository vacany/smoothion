import os

import numpy as np
import pandas as pd
from datatools.lidar_dataset import Lidar_Dataset
from datatools.visualizer.vis import visualize_points3D, visualize_poses
from datatools.stats.segmentation import IOU



def generate_gt_ids(dataset):
       for sequence in range(11):
           for frame_id in range(dataset.data_info[sequence]['nbr_of_frames']):
                  print(frame_id)

                  save_output_folder = dataset.data_info[sequence]['path'] + '/gt_ids_tracking_labels'
                  os.makedirs(save_output_folder, exist_ok=True)

                  data = dataset.get_sequence(sequence, frame_id, frame_id, dtypes=['new_instance_label'])

                  gt_instances = data['new_instance_label']
                  gt_instances[gt_instances == -1] = 0

                  # assert ((self.sem_label + (self.inst_label << 16) == label).all())

                  save_labels = open(f'{save_output_folder}/{frame_id:06d}label.label', 'wb')
                  save_labels.write(gt_instances.astype(np.uint32))# + (gt_instances << 16))
                  save_labels.close()


dataset = Lidar_Dataset('semantic_kitti')

all_metric = IOU(2, clazz_names=['Static', 'Dynamic'])
tmp_metric = IOU(2, clazz_names=['Static', 'Dynamic'])

stat_array = []

np.set_printoptions(2)
pd.set_option('display.precision', 2)
for sequence in list(range(8)) + [9,10]:

       min_f, max_f = 0, dataset.data_info[sequence]['nbr_of_frames']


       mos_labels = []
       track_list = []
       ego_labels = []
       dyn_labels = []
       for i in range(min_f, max_f):
              # print(i)
              ego = np.load(f'/mnt/personal/vacekpa2/data/semantic_kitti/dataset/sequences/{sequence:02d}/ego_dynamic_points/{i:06d}.npy')
              dyn_label = np.load(f'/mnt/personal/vacekpa2/data/semantic_kitti/dataset/sequences/{sequence:02d}/dynamic_label/{i:06d}.npy')
              label = np.fromfile(f'/mnt/personal/vacekpa2/data/semantic_kitti/dataset/gt_mos/sequences/{sequence:02}/predictions/{i:06d}.label', np.uint32).reshape((-1))

              sem_label = label & 0xFFFF
              inst = label >> 16

              mos_labels.append(sem_label)
              ego_labels.append(ego)
              dyn_labels.append(dyn_label)
              # track_list.append(track & 0xFFFF)

       mos_labels = np.concatenate(mos_labels) # 9 - static; 251 - dynamic
       ego_labels = np.concatenate(ego_labels)
       dyn_labels = np.concatenate(dyn_labels)
       # tracks = np.concatenate(track_list)


       corrected_labels = ego_labels.copy()
       corrected_labels[mos_labels == 9] = 0     # where static were discriminated

       print(f"Sequence {sequence} --- ")
       tmp_metric.update(dyn_labels, corrected_labels)
       precs, recals, ious = tmp_metric.return_stats()

       stat_array.append(np.array((precs[1], recals[1], ious[1])))
       # generate table
       tmp_metric.print_stats()
       tmp_metric.reset()

       all_metric.update(dyn_labels, corrected_labels)

print(f"All Sequences Together --- ")
all_metric.print_stats()
precs, recals, ious = all_metric.return_stats()
stat_array.append(np.array((precs[1], recals[1], ious[1])))

stat_array = np.stack(stat_array).T
df = pd.DataFrame(stat_array * 100, index=['Precision', 'Recall', 'IoU'], columns=list(range(8)) + [9,10,'All'])

print(df.to_latex())
print(df)


