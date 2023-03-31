import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datatools.lidar_dataset import Lidar_Dataset

class Motion_Sequence():
    def __init__(self, sequence, dataset_type='semantic_kitti', start=0, end=np.inf):
        self.sequence = sequence
        self.dataset_type = dataset_type
        self.dataset = Lidar_Dataset(dataset_type)

        self.start = start

        max_seq_fr = self.dataset.prep_dataset.data_info[self.sequence]['nbr_of_frames']
        if end > max_seq_fr:
            self.end = max_seq_fr

        else:
            self.end = end



    def get_raw_data(self):
        data = self.dataset.prep_dataset.get_base_data(sequence=self.sequence, start=self.start, end=self.end)

        return data

    def get_specific_data(self, data_type, form='list'):
        sequence_path = self.dataset.prep_dataset.sequence_paths[self.sequence]

        data_paths = sorted(glob.glob(sequence_path + f'/{data_type}*'))[self.start: self.end]

        if form == 'list':
            return [np.load(file) for file in data_paths]

        elif form == 'concat':
            return np.concatenate([np.load(file) for file in data_paths])

        else:
            raise NotImplemented("Form of data postprocess is not implemented")

