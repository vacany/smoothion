import importlib
import os

import numpy as np
import torch
import multiprocessing

from my_datasets import paths

from torch.utils.data._utils.collate import default_collate

def remove_out_of_bounds_points(pc, x_min, x_max, y_min, y_max, z_min, z_max):
    # Max needs to be exclusive because the last grid cell on each axis contains
    # [((grid_size - 1) * cell_size) + *_min, *_max).
    #   E.g grid_size=512, cell_size = 170/512 with min=-85 and max=85
    # For z-axis this is not necessary, but we do it for consistency
    mask = (pc[:, 0] >= x_min) & (pc[:, 0] < x_max) \
           & (pc[:, 1] >= y_min) & (pc[:, 1] < y_max) \
           & (pc[:, 2] >= z_min) & (pc[:, 2] < z_max)
    pc_valid = pc[mask]
    y_valid = mask

    return pc_valid, y_valid


def create_pillars_matrix(pc_valid, grid_cell_size, x_min, y_min, z_min, z_max, n_pillars_x):
    """
    Compute the pillars using matrix operations.
    :param pc: point cloud data. (N_points, features) with the first 3 features being the x,y,z coordinates.
    :return: augmented_pointcloud, grid_cell_indices, y_valid
    """
    # with torch no grad
    num_laser_features = pc_valid.shape[1] - 3  # Calculate the number of laser features that are not the coordinates.

    # Calculate the cell id that this entry falls into
    # Store the X, Y indices of the grid cells for each point cloud point
    grid_cell_indices = np.zeros((pc_valid.shape[0], 2), dtype=int)
    grid_cell_indices[:, 0] = ((pc_valid[:, 0] - x_min) / grid_cell_size).astype(int)
    grid_cell_indices[:, 1] = ((pc_valid[:, 1] - y_min) / grid_cell_size).astype(int)

    # Initialize the new pointcloud with 8 features for each point
    augmented_pc = np.zeros((pc_valid.shape[0], 6 + num_laser_features))
    # Set every cell z-center to the same z-center
    augmented_pc[:, 2] = z_min + ((z_max - z_min) * 1 / 2)
    # Set the x cell center depending on the x cell id of each point
    augmented_pc[:, 0] = x_min + 1 / 2 * grid_cell_size + grid_cell_size * grid_cell_indices[:, 0]
    # Set the y cell center depending on the y cell id of each point
    augmented_pc[:, 1] = y_min + 1 / 2 * grid_cell_size + grid_cell_size * grid_cell_indices[:, 1]

    # Calculate the distance of the point to the center.
    # x
    augmented_pc[:, 3] = pc_valid[:, 0] - augmented_pc[:, 0]
    # y
    augmented_pc[:, 4] = pc_valid[:, 1] - augmented_pc[:, 1]
    # z
    augmented_pc[:, 5] = pc_valid[:, 2] - augmented_pc[:, 2]

    # Take the two laser features
    augmented_pc[:, 6:] = pc_valid[:, 3:]
    # augmented_pc = [cx, cy, cz,  Δx, Δy, Δz, l0, l1]

    # Convert the 2D grid indices into a 1D encoding
    # This 1D encoding is used by the models instead of the more complex 2D x,y encoding
    # To make things easier we transform the 2D indices into 1D indices
    # The cells are encoded as j = x * grid_width + y and thus give an unique encoding for each cell
    # E.g. if we have 512 cells in both directions and x=1, y=2 is encoded as 512 + 2 = 514.
    # Each new row of the grid (x-axis) starts at j % 512 = 0.
    grid_cell_indices = grid_cell_indices[:, 0] * n_pillars_x + grid_cell_indices[:, 1]

    return augmented_pc, grid_cell_indices





class ApplyPillarization:
    def __init__(self, grid_cell_size, x_min, y_min, z_min, z_max, n_pillars_x):
        self._grid_cell_size = grid_cell_size
        self._z_max = z_max
        self._z_min = z_min
        self._y_min = y_min
        self._x_min = x_min
        self._n_pillars_x = n_pillars_x

    """ Transforms an point cloud to the augmented pointcloud depending on Pillarization """

    def __call__(self, x):
        point_cloud, grid_indices = create_pillars_matrix(x,
                                                          grid_cell_size=self._grid_cell_size,
                                                          x_min=self._x_min,
                                                          y_min=self._y_min,
                                                          z_min=self._z_min, z_max=self._z_max,
                                                          n_pillars_x=self._n_pillars_x)
        return point_cloud, grid_indices


def drop_points_function(x_min, x_max, y_min, y_max, z_min, z_max):
    def inner(x, y):
        return remove_out_of_bounds_points(x, y,
                                           x_min=x_min,
                                           y_min=y_min,
                                           z_min=z_min,
                                           z_max=z_max,
                                           x_max=x_max,
                                           y_max=y_max
                                           )

    return inner

# ------------- Preprocessing Functions ---------------


def get_coordinates_and_features(point_cloud, transform=None):
    """
    Parse a point clound into coordinates and features.
    :param point_cloud: Full [N, 9] point cloud
    :param transform: Optional parameter. Transformation matrix to apply
    to the coordinates of the point cloud
    :return: [N, 5] where N is the number of points and 5 is [x, y, z, intensity, elongation]
    """
    points_coord, features, flows = point_cloud[:, 0:3], point_cloud[:, 3:5], point_cloud[:, 5:]
    if transform is not None:
        ones = np.ones((points_coord.shape[0], 1))
        points_coord = np.hstack((points_coord, ones))
        points_coord = transform @ points_coord.T
        points_coord = points_coord[0:-1, :]
        points_coord = points_coord.T
    point_cloud = np.hstack((points_coord, features))
    return point_cloud


def _pad_batch(batch, nbr_pts_batch):
    # Get the number of points in the largest point cloud
    true_number_of_points = [e[0].shape[0] for e in batch]
    max_points_prev = np.max(true_number_of_points + [e[0].shape[0] for e in nbr_pts_batch])


    # We need a mask of all the points that actually exist
    zeros = np.zeros((len(batch), max_points_prev), dtype=bool)
    # Mark all points that ARE NOT padded
    for i, n in enumerate(true_number_of_points):
        zeros[i, :n] = 1

    # resize all tensors to the max points size
    # Use np.pad to perform this action. Do not pad the second dimension and pad the first dimension AFTER only


    return [
        [np.pad(entry[0], ((0, max_points_prev - entry[0].shape[0]), (0, 0))),
         np.pad(entry[1], (0, max_points_prev - entry[1].shape[0])) if entry[1] is not None else np.empty(shape=(max_points_prev, )),
         zeros[i],
         # np.pad(entry[2], (0, max_points_prev - entry[1].shape[0]), constant_values=-1), #if entry[3] is not None else np.empty(shape=(max_points_prev,)),
         entry[2],  #
         np.pad(entry[3], (0, max_points_prev - entry[3].shape[0]), constant_values=-1),
         np.pad(entry[4], ((0, max_points_prev - entry[4].shape[0]), (0, 0)), constant_values=-1),
         # entry[5]
         ] for i, entry in enumerate(batch)
    ]


def _pad_targets(batch):
    true_number_of_points = [e.shape[0] for e in batch]
    max_points = np.max(true_number_of_points)
    return [
        np.pad(entry, ((0, max_points - entry.shape[0]), (0, 0)))
        for entry in batch
    ]


def custom_collate_batch(batch):
    """
    This version of the collate function create the batch necessary for the input to the network.

    Take the list of entries and batch them together.
        This means a batch of the previous images and a batch of the current images and a batch of flows.
    Because point clouds have different number of points the batching needs the points clouds with less points
        being zero padded.
    Note that this requires to filter out the zero padded points later on.

    :param batch: batch_size long list of ((prev, cur), flows) pointcloud tuples with flows.
        prev and cur are tuples of (point_cloud, grid_indices, mask)
         point clouds are (N_points, features) with different N_points each
    :return: ((batch_prev, batch_cur), batch_flows)
    """
    # Build numpy array with data

    # Only convert the points clouds from numpy arrays to tensors
    # entry[0, 0] is the previous (point_cloud, grid_index) entry
    batch_previous = [entry[0] for entry in batch]
    batch_current = [entry[1] for entry in batch]

    batch_previous = _pad_batch(batch_previous, batch_current)
    batch_current = _pad_batch(batch_current, batch_previous)


    # For the targets we can only transform each entry to a tensor and not stack them
    # batch_targets = [
    #     entry[1] for entry in batch
    # ]
    # batch_targets = _pad_targets(batch_targets)

    # Call the default collate to stack everything
    batch_previous = default_collate(batch_previous) # for collating with same pts nbr
    batch_current = default_collate(batch_current)
    # batch_ego = default_collate(batch_ego)
    # batch_targets = default_collate(batch_targets)

    # Return a tensor that consists of
    # the data batches consist of batches of tensors
    #   1. (batch_size, max_n_points, features) the point cloud batch
    #   2. (batch_size, max_n_points) the 1D grid_indices encoding to map to
    #   3. (batch_size, max_n_points) the 0-1 encoding if the element is padded
    # Batch previous for the previous frame
    # Batch current for the current frame

    # The targets consist of
    #   (batch_size, max_n_points, target_features). should by 4D x,y,z flow and class id

    return (batch_previous, batch_current) #, batch_targets


class SceneFlowLoader(torch.utils.data.Dataset):


    def __init__(self, name_of_dataset, cfg):
        self.name_of_dataset = name_of_dataset
        self.cfg = cfg
        self.apply_transformation = True
        grid_cell_size = (cfg['x_max'] + abs(cfg['x_min'])) / cfg['grid_size']

        n_pillars_x = cfg['grid_size']
        # n_pillars_y = cfg['grid_size']

        # This can be later split in datasetclass
        self.pilarization = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=cfg['x_min'], y_min=cfg['y_min'],
                                          z_min=cfg['z_min'], z_max=cfg['z_max'], n_pillars_x=n_pillars_x,
                                          )

        self.choose_dataset(name_of_dataset)

    def collect_data(self):
        info = self.sequence.info()

        nbr_of_seqs = info['nbr_of_seqs']

        from my_datasets.kitti.semantic_kitti import SemanticKitti_Sequence

        data_files = []

        all_seqs = [SemanticKitti_Sequence(seq) for seq in range(nbr_of_seqs)]

        frames_of_seqs = [[curr_seq.sequence_nbr, list(range(len(curr_seq)))] for curr_seq in all_seqs]

        # do it better with respect to training splits!

        pass

    def __len__(self):
        return len(self.all_indices) # to use the last current label as well

    def preproces_data(self, pts1, pts2, ego_label1, ego_label2, frame_id):
        # first eliminate out of boundaries
        pts1, mask1 = remove_out_of_bounds_points(pts1, self.cfg['x_min'] + 0.2, self.cfg['x_max'] - 0.2,
                                                  self.cfg['y_min'] + 0.2,
                                                  self.cfg['y_max'] - 0.2,
                                                  self.cfg['z_min'], self.cfg['z_max'])
        pts2, mask2 = remove_out_of_bounds_points(pts2, self.cfg['x_min'], self.cfg['x_max'], self.cfg['y_min'],
                                                  self.cfg['y_max'],
                                                  self.cfg['z_min'], self.cfg['z_max'])  # add here the AV points

        ego_label1 = ego_label1[mask1]
        ego_label2 = ego_label2[mask2]


        # Subsample to get previous and current the same number of points
        min_nbr_pts = np.min([pts1.shape[0], pts2.shape[0]])

        subsample_idx1 = np.arange(min_nbr_pts)
        subsample_idx2 = np.arange(min_nbr_pts)

        pts1 = pts1[subsample_idx1]
        pts2 = pts2[subsample_idx2]

        ego_label1 = ego_label1[subsample_idx1]
        ego_label2 = ego_label2[subsample_idx2]

        # this can be changed in config - elongation
        pts1 = np.insert(pts1, 4, 1, axis=1)
        pts2 = np.insert(pts2, 4, 1, axis=1)

        # dynamic has priority
        cell_size = np.abs(2 * self.cfg['x_min'] / self.cfg['grid_size'])
        label_grid2 = - np.ones((self.cfg['grid_size'], self.cfg['grid_size']), dtype=int)
        label_ind = ((pts2[:, :2] - (self.cfg['x_min'], self.cfg['y_min'])) / cell_size).astype(int)

        label_grid2[label_ind[ego_label2 == 0, 0], label_ind[ego_label2 == 0, 1]] = 0
        label_grid2[label_ind[ego_label2 == 1, 0], label_ind[ego_label2 == 1, 1]] = 1

        label_grid1 = - np.ones((self.cfg['grid_size'], self.cfg['grid_size']), dtype=int)

        pts1, grid1 = self.pilarization(pts1)
        pts2, grid2 = self.pilarization(pts2)

        mask1 = np.ones(pts1[:, 0].shape, dtype=bool)
        mask2 = np.ones(pts2[:, 0].shape, dtype=bool)  # is right?


        prev_batch = (pts1, grid1, mask1, ego_label1, label_grid1,  frame_id)
        current_batch = (pts2, grid2, mask2, ego_label2, label_grid2, frame_id)  # normal pts as well?

        x = (prev_batch, current_batch)  # is this really the pts1 and pts2?

        return x

    def __getitem__(self, idx):

        seq_id, frame = self.all_indices[idx]
        current_sequence = self.sequence_list[seq_id]

        pts1, pts2 = current_sequence.get_two_synchronized_frames(frame, frame+1, pts_source='lidar')

        # tmp unifing the points for same road removal
        pts1[:, 2] -= current_sequence.ground_ego_height
        pts2[:, 2] -= current_sequence.ground_ego_height


        pts1, mask1 = remove_out_of_bounds_points(pts1, self.cfg['x_min'] + self.cfg['cell_size'], self.cfg['x_max'] - self.cfg['cell_size'],
                                                  self.cfg['y_min'] + self.cfg['cell_size'], self.cfg['y_max'] - self.cfg['cell_size'],
                                                  self.cfg['z_min'], self.cfg['z_max'])
        pts2, mask2 = remove_out_of_bounds_points(pts2, self.cfg['x_min'] + self.cfg['cell_size'], self.cfg['x_max'] - self.cfg['cell_size'],
                                                  self.cfg['y_min'] + self.cfg['cell_size'], self.cfg['y_max'] - self.cfg['cell_size'],
                                                  self.cfg['z_min'], self.cfg['z_max'])  # add here the AV points

        target1 = current_sequence.get_feature(frame, name=self.label_source)[mask1]
        target2 = current_sequence.get_feature(frame+1, name=self.label_source)[mask2]

        dynamic_label1 = current_sequence.get_feature(frame, name='dynamic_label')[mask1].astype('int32')
        dynamic_label2 = current_sequence.get_feature(frame+1, name='dynamic_label')[mask2].astype('int32')

        flow1 = current_sequence.get_feature(frame, name='flow_label')[mask1]
        flow2 = current_sequence.get_feature(frame + 1, name='flow_label')[mask2]

        target_grid1 = np.zeros((self.cfg['grid_size'], self.cfg['grid_size']), dtype=int)
        target_grid2 = np.zeros((self.cfg['grid_size'], self.cfg['grid_size']), dtype=int)

        target_ind1 = ((pts1[:, :2] - (self.cfg['x_min'], self.cfg['y_min'])) / self.cfg['cell_size']).astype(int)
        target_ind2 = ((pts2[:, :2] - (self.cfg['x_min'], self.cfg['y_min'])) / self.cfg['cell_size']).astype(int)

        target_grid1[target_ind1[target1 == 0, 0], target_ind1[target1 == 0, 1]] = 1
        target_grid1[target_ind1[target1 == 1, 0], target_ind1[target1 == 1, 1]] = 2

        target_grid2[target_ind2[target2 == 0, 0], target_ind2[target2 == 0, 1]] = 1
        target_grid2[target_ind2[target2 == 1, 0], target_ind2[target2 == 1, 1]] = 2

        prev = (pts1, target1, target_grid1, dynamic_label1, flow1)
        curr = (pts2, target2, target_grid2, dynamic_label2, flow2)


        # x = self.preproces_data(pts1, pts2, ego_label1, ego_label2, idx)
        x = [prev, curr]

        return x

    def create_pillar_batch_gpu(self, pts, mask):
        """
        Compute the pillars using matrix operations.
        :param pc: point cloud data. (N_points, features) with the first 3 features being the x,y,z coordinates.
        :return: augmented_pointcloud, grid_cell_indices, y_valid
        """
        BS = len(pts)
        # bs_ind = torch.cat([bs_idx * torch.ones(pts.shape[1], dtype=torch.long, device=pts.device) for bs_idx in range(BS)])

        # with torch no grad
        num_laser_features = self.cfg['point_features'] - 6  # Calculate the number of laser features that are not the coordinates.
        # num_laser_features = pts.shape[-1] - 3  # Calculate the number of laser features that are not the coordinates.

        # Calculate the cell id that this entry falls into
        # Store the X, Y indices of the grid cells for each point cloud point
        coor_shift = torch.tile(torch.tensor((self.cfg['x_min'], self.cfg['y_min']), dtype=torch.float, device=pts.device), dims=(BS, 1, 1))


        # grid_cell_indices = torch.zeros((pts.shape[0], 2), dtype=torch.long, device=pts.device)
        grid_cell_indices = ((pts[..., :2] - coor_shift) / self.cfg['cell_size']).long()
        # grid_cell_indices[:, 0] = ((pc_valid[:, 0] - self.cfg['x_min']) / self.cfg['cell_size']).astype(int)
        # grid_cell_indices[:, 1] = ((pc_valid[:, 1] - self.cfg['y_min']) / self.cfg['cell_size']).astype(int)


        # Initialize the new pointcloud with 8 features for each point
        augmented_pc = torch.zeros((pts.shape[0], pts.shape[1], 6 + num_laser_features), device=pts.device)
        # Set every cell z-center to the same z-center
        augmented_pc[..., 2] = self.cfg['z_min'] + ((self.cfg['z_max'] - self.cfg['z_min']) * 1 / 2)
        # Set the x cell center depending on the x cell id of each point
        augmented_pc[..., 0] = self.cfg['x_min'] + 1 / 2 * self.cfg['cell_size'] + self.cfg['cell_size'] * grid_cell_indices[..., 0]
        # Set the y cell center depending on the y cell id of each point
        augmented_pc[..., 1] = self.cfg['y_min'] + 1 / 2 * self.cfg['cell_size'] + self.cfg['cell_size'] * grid_cell_indices[..., 1]

        # Calculate the distance of the point to the center.
        # x
        augmented_pc[..., 3] = pts[..., 0] - augmented_pc[..., 0]
        # y
        augmented_pc[..., 4] = pts[..., 1] - augmented_pc[..., 1]
        # z
        augmented_pc[..., 5] = pts[..., 2] - augmented_pc[..., 2]

        # Take the two laser features
        if self.cfg['point_features'] > 6:
            augmented_pc[..., 6:] = pts[..., 3:]
        # augmented_pc = [cx, cy, cz,  Δx, Δy, Δz, l0, l1]

        # Convert the 2D grid indices into a 1D encoding
        # This 1D encoding is used by the models instead of the more complex 2D x,y encoding
        # To make things easier we transform the 2D indices into 1D indices
        # The cells are encoded as j = x * grid_width + y and thus give an unique encoding for each cell
        # E.g. if we have 512 cells in both directions and x=1, y=2 is encoded as 512 + 2 = 514.
        # Each new row of the grid (x-axis) starts at j % 512 = 0.
        grid_cell_indices = grid_cell_indices[..., 0] * self.cfg['grid_size'] + grid_cell_indices[..., 1]

        # mask
        return augmented_pc, grid_cell_indices, mask

    # def construct_batched_cuda_grid(pts, feature, cfg, device):
    #     '''
    #     Assumes BS x N x CH (all frames same number of fake pts with zeros in the center)
    #     :param pts:
    #     :param feature:
    #     :param cfg:
    #     :return:
    #     '''
    #     BS = len(pts)
    #     bs_ind = torch.cat([bs_idx * torch.ones(pts.shape[1], dtype=torch.long, device=device) for bs_idx in range(BS)])
    #
    #     feature_grid = - torch.ones(BS, cfg['grid_size'], cfg['grid_size'], device=device).long()
    #
    #     cell_size = np.abs(2 * cfg['x_min'] / cfg['grid_size'])
    #
    #     coor_shift = torch.tile(torch.tensor((cfg['x_min'], cfg['y_min']), dtype=torch.float, device=device),
    #                             dims=(BS, 1, 1))
    #
    #     feature_ind = ((pts[:, :, :2] - coor_shift) / cell_size).long()
    #
    #     feature_grid[bs_ind, feature_ind.flatten(0, 1)[:, 0], feature_ind.flatten(0, 1)[:, 1]] = feature.flatten()
    #
    #     return feature_grid

    def return_dataloader(self, batch_size=1, num_workers=0, shuffle=True):
        cpu_count = multiprocessing.cpu_count()

        # num_workers = np.min((cpu_count, self.cfg['BS'])) if self.cfg["BS"] > 1 else 0

        dataloader = torch.utils.data.DataLoader(self, batch_size=self.cfg['BS'], shuffle=shuffle,
                                                 num_workers=num_workers, collate_fn=custom_collate_batch)

        return dataloader

    def calculate_CE_weights(self):
        dyn, stat = 0, 0
        choose_nbr = 100
        weights = (1., 1.)

        chosen_indices = np.random.randint(0, len(self.all_indices), choose_nbr)

        cell_size = np.abs(2 * self.cfg['x_min'] / self.cfg['grid_size'])

        for _, coor in enumerate(np.array(self.all_indices)[chosen_indices]):

            current_sequence = self.sequence_list[coor[0]]

            pts2 = current_sequence.get_feature(coor[1] + 1, name='lidar')  # for prev frame    # tmp dynamic label
            ego_label2 = current_sequence.get_feature(coor[1] + 1, name=self.label_source)  # for prev frame    # tmp dynamic label

            pts2, mask2 = remove_out_of_bounds_points(pts2, self.cfg['x_min'] + 0.2, self.cfg['x_max'] - 0.2,
                                                      self.cfg['y_min'] + 0.2,
                                                      self.cfg['y_max'] - 0.2,
                                                      self.cfg['z_min'], self.cfg['z_max'])

            ego_label2 = ego_label2[mask2]
            label_grid2 = - np.ones((self.cfg['grid_size'], self.cfg['grid_size']), dtype=int)
            label_ind = ((pts2[:, :2] - (self.cfg['x_min'], self.cfg['y_min'])) / cell_size).astype(int)

            label_grid2[label_ind[ego_label2 == 0, 0], label_ind[ego_label2 == 0, 1]] = 0
            label_grid2[label_ind[ego_label2 == 1, 0], label_ind[ego_label2 == 1, 1]] = 1

            stat += np.sum(label_grid2 == 0)
            dyn += np.sum(label_grid2 == 1)
            weights = (dyn / stat, stat / dyn)
            # print(_, weights)

        return weights

    def choose_dataset(self, name_of_dataset):
        ''' Name of the dataset refers to the variable name in my_datasets.paths
        '''

        dataset_meta = getattr(paths, name_of_dataset.upper())
        # later preload whole sequences
        dataset_class = dataset_meta['dataset_class']
        self.label_source = dataset_meta['label_source']
        self.used_sequences = dataset_meta['used_seqs']

        sequence = dataset_class(0)

        self.dataset_info = sequence.info()
        self.nbr_of_sequences = self.dataset_info['nbr_of_seqs']

        self.sequence_frames_dict = {seq_id: len(dataset_class(seq_id)) for seq_id in range(self.nbr_of_sequences) if seq_id in self.used_sequences}
        self.sequence_list = [dataset_class(seq_id) for seq_id in range(self.nbr_of_sequences)]# if seq_id in self.used_sequences]
        self.all_indices = []


        for k, v in self.sequence_frames_dict.items():
            for i in range(v - 1):  # - 1 because of last invalid flow
                self.all_indices.append((k, i))

        return dataset_class

