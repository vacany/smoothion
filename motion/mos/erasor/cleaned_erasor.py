import numpy as np
import os
import time
import yaml
from sklearn.decomposition import PCA

from datatools.visualizer.vis import *
from datatools.lidar_dataset import Lidar_Dataset
from datatools.structures.bev import BEV

DATA_DIR = os.environ["HOME"] + '/data/'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print(f"{bcolors.OKBLUE}{method.__name__}{bcolors.ENDC} elapsed in {bcolors.OKBLUE}{te - ts:.3f} seconds {bcolors.ENDC}")
        return result

    return timed

def erasor_config():
    config = {  "h_max" : 1.2,
                "h_min" : -3.0,  # must be below ground
                "L_max" : 80.0,
                "NR" : 20, # 15
                "NS" : 108, # 60
                "scan_ratio_test_threshold" : 0.2,
                "minimum_nbr_pts" : 10,
                "tau_seed_height" : 0.5,
                "ground_plane_iterations" : 3,
                "cell_size" : 0.1,
                'tau_ground_margin' : 0.25,
    }

    return config


class Erasor():
    def __init__(self, output_dir, **kwargs):
        self.output_dir = output_dir
        self.cfg = erasor_config()
        for k, v in kwargs.items():
            self.cfg[k] = v

    def preload_data(self, pcls, poses, pcl_size):
        self.pcls = pcls
        self.poses = poses
        self.pcl_size = pcl_size

        self.points_m = np.concatenate(pcls)[:, :3]
        self.all_dynamic = np.zeros(self.points_m.shape[0], dtype=bool)
        self.all_ground = np.zeros(self.points_m.shape[0], dtype=bool)

        self.raster_points = self.get_sector_points()
        os.makedirs(self.output_dir + f'/dynamic/', exist_ok=True)
        os.makedirs(self.output_dir + f'/ground/', exist_ok=True)

        with open(self.output_dir + '/cfg.yml', 'w') as f:
            yaml.dump(self.cfg, stream=f)




    def inverse_pose_transform(self, points, pose):
        origin_points = (np.insert(points, obj=3, values=1, axis=1) @ np.linalg.inv(pose.T))[:, :3]

        return origin_points

    def crop_volume_of_interest(self, points, pose, koef=1.):
        x, y, z = pose[:3, -1]

        mask_x = (points[:, 0] < x + self.cfg['L_max'] * koef) & (points[:, 0] > x - self.cfg['L_max'] * koef)
        mask_y = (points[:, 1] < y + self.cfg['L_max'] * koef) & (points[:, 1] > y - self.cfg['L_max'] * koef)
        mask_z = (points[:, 2] < z + self.cfg['h_max']) & (points[:, 2] > z + self.cfg['h_min'])

        mask = mask_x * mask_y * mask_z

        return mask


    def calculate_angle_and_radius(self, points):
        angle = np.arctan2(points[:, 1], points[:, 0])
        radius = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

        return angle, radius

    def calculate_sector_indices(self, angle, radius):
        i_idx = - np.ones(radius.shape)
        j_idx = - np.ones(angle.shape)

        for i in range(1, self.cfg['NR'] + 1):
            radius_mask = (((i - 1) * self.cfg['L_max'] / self.cfg['NR']) <= radius) & (radius <= (i * self.cfg['L_max'] / self.cfg['NR']))
            i_idx[radius_mask] = i - 1

        for j in range(1, self.cfg['NS'] + 1):
            angle_mask = (((j - 1) * 2 * np.pi / self.cfg['NS']) - np.pi <= angle) & (angle <= (j * 2 * np.pi / self.cfg['NS']) - np.pi)
            j_idx[angle_mask] = j - 1

        return i_idx, j_idx

    @timeit
    def prepare_frame_data(self, frame_id):

        self.pose = self.poses[frame_id]

        self.points_q = self.pcls[frame_id][:, :3].copy()

        self.volume_q = Er.crop_volume_of_interest(self.points_q, self.pose, koef=1.5)
        self.volume_q_idx = np.argwhere(self.volume_q)[:,0]
        self.vis_points_q = self.points_q[self.volume_q].copy()
        self.points_q = (np.insert(self.points_q[self.volume_q], obj=3, values=1, axis=1) @ np.linalg.inv(self.pose.T))[:, :3]
        self.volume_q_new = Er.crop_volume_of_interest(self.points_q, np.eye(4))
        self.volume_q[self.volume_q_idx] = self.volume_q_new

        self.vis_points_q = self.vis_points_q[self.volume_q_new]
        self.points_q = self.points_q[self.volume_q_new]


        # Mapgen
        self.map_valid_mask = np.zeros(self.points_m.shape[0], dtype=bool)
        self.volume_mask = Er.crop_volume_of_interest(self.points_m, self.pose, koef=1.5).copy()
        self.map_valid_mask[self.volume_mask] = True
        self.map_valid_mask[self.all_dynamic] = False

        self.volume_idx = np.argwhere(self.map_valid_mask)[:, 0]  # index for mapping voi_points_m back to points_m

        origin_points = (np.insert(self.points_m[self.map_valid_mask], obj=3, values=1, axis=1) @ np.linalg.inv(self.pose.T))[:, :3]

        new_volume = Er.crop_volume_of_interest(origin_points, pose=np.eye(4))  # already transformed cropping
        # Connect these two!
        self.map_valid_mask[self.volume_idx] = new_volume
        self.voi_points_m = origin_points[new_volume]


    def get_sector_points(self):
        nx, ny = (self.cfg['L_max'] + 10, self.cfg['L_max'] + 10)
        x = np.linspace(-nx, nx, int(2 * nx / self.cfg['cell_size'] + 1))
        y = np.linspace(-nx, ny, int(2 * ny / self.cfg['cell_size'] + 1))

        xv, yv = np.meshgrid(x, y)
        raster_points = np.stack((np.concatenate(xv), np.concatenate(yv), np.ones(xv.shape).reshape(-1),
                                  np.ones(xv.shape).reshape(-1))).T

        # properties of points
        angles, radius = self.calculate_angle_and_radius(raster_points)

        i_idx, j_idx = self.calculate_sector_indices(angles, radius)

        j_idx[i_idx == -1] = -1

        raster_points[:, 2] = i_idx
        raster_points[:, 3] = j_idx

        return raster_points


    def visualize_prepared_data(self):
        visualize_multiple_pcls(self.voi_points_m, self.raster_points[:,[0,1,3]], self.points_q)

    def transfer_sectors_to_points(self, points):
        Bev = BEV((self.cfg['cell_size'], self.cfg['cell_size']))
        Bev.create_bev_template_from_points(self.raster_points)
        radius_img = Bev.generate_bev(self.raster_points[:, :2], self.raster_points[:, 2])
        sector_img = Bev.generate_bev(self.raster_points[:, :2], self.raster_points[:, 3])

        N_r = Bev.transfer_features_to_points(points, radius_img)
        N_s = Bev.transfer_features_to_points(points, sector_img)

        return N_r, N_s

    @timeit
    def calculate_heights_sector(self, points, N_r, N_s):
        Sector_bev = BEV((1, 1))
        sector_coors = np.stack((N_r, N_s, np.ones(N_r.shape[0]))).T
        bev_temp_xyz = np.stack((np.array((-1, 0, 0)),
                                 np.array((-1, self.cfg['NS'] - 1, 0)),
                                 np.array((self.cfg['NR'] - 1, -1, 0)),
                                 np.array((self.cfg['NR'] - 1, self.cfg['NS'] - 1, 1))))

        Sector_bev.create_bev_template_from_points(bev_temp_xyz)  # TOdo will be robust

        from_min_h_idx = points[:, 2].argsort()
        from_max_h_idx = points[:, 2].argsort()[::-1]

        max_h_grid = Sector_bev.generate_bev(sector_coors[from_min_h_idx], points[:, 2][from_min_h_idx])
        min_h_grid = Sector_bev.generate_bev(sector_coors[from_max_h_idx], points[:, 2][from_max_h_idx])

        delta_h_grid = max_h_grid - min_h_grid
        vis_delta_h_grid = delta_h_grid.copy()
        delta_h_grid = delta_h_grid[1:, 1:]  # without -1 index in radius and sector

        # for Visualization
        max_h_point = Sector_bev.transfer_features_to_points(sector_coors, max_h_grid)
        min_h_point = Sector_bev.transfer_features_to_points(sector_coors, min_h_grid)
        delta_h_point = Sector_bev.transfer_features_to_points(sector_coors, vis_delta_h_grid)


        # visualize_points3D(points, min_h_point)
        # visualize_points3D(points, max_h_point)
        # visualize_points3D(points, delta_h_point)   # checked

        return delta_h_grid  # delta_h_point,, max_h_point, min_h_point

    @timeit
    def calculate_scan_ratio(self, delta_h_grid_m, delta_h_grid_q):
        # scan ratio == 1 is for static
        scan_ratio_grid = np.ones(delta_h_grid_m.shape)

        # Validation mask for bins
        valid_operation_mask = (delta_h_grid_q != 0.) & (delta_h_grid_m != 0.)
        scan_ratio_grid[valid_operation_mask] = np.min((
                delta_h_grid_m[valid_operation_mask] / delta_h_grid_q[valid_operation_mask],
                delta_h_grid_q[valid_operation_mask] / delta_h_grid_m[valid_operation_mask]),
                axis=0)

        Sector_bev = BEV((1, 1))
        valid_raster_mask = np.all(self.raster_points[:, 2:] != -1, axis=1)
        Sector_bev.create_bev_template_from_points(self.raster_points[valid_raster_mask, 2:])
        sector_coors = np.stack((self.N_r_M, self.N_s_M, np.ones(self.N_r_M.shape[0]))).T
        # sector_coors_q = np.stack((self.N_r_q, self.N_s_q, np.ones(self.N_r_q.shape[0]))).T
        # reimplementation of C++ code
        map_bigger = delta_h_grid_m >= delta_h_grid_q
        above_ground = delta_h_grid_m > 0.2 # slightly above ground

        # mask = map_bigger & above_ground & valid_operation_mask
        mask = map_bigger & above_ground & valid_operation_mask & (scan_ratio_grid < 0.2)

        scan_ratio_grid[mask==False] = 1

        dyn_props = Sector_bev.transfer_features_to_points(sector_coors, scan_ratio_grid)


        # without -1 index
        # min points check in both M and q
        # Backprop from 2D grid for speed, creates bin coordinates
        histo = np.histogram2d(self.N_s_M, self.N_r_M, range=[[0, self.cfg['NS']], [0, self.cfg['NR']]], bins=(self.cfg['NS'], self.cfg["NR"]))
        histo_q = np.histogram2d(self.N_s_q, self.N_r_q, range=[[0, self.cfg['NS']], [0, self.cfg['NR']]], bins=(self.cfg['NS'], self.cfg["NR"]))



        # count_valid_mask = histo[0] > self.cfg['minimum_nbr_pts']
        # count_valid_mask_q = histo_q[0] > self.cfg['minimum_nbr_pts']



        self.histo_pts = Sector_bev.transfer_features_to_points(sector_coors, histo[0].T)
        self.histo_pts_q = Sector_bev.transfer_features_to_points(sector_coors, histo_q[0].T)
        min_h_q_pts = Sector_bev.transfer_features_to_points(sector_coors, delta_h_grid_q)
        self.min_h_M_pts = Sector_bev.transfer_features_to_points(sector_coors, delta_h_grid_m)

        # calculation of paper equation - static deterministically and cleaned
        grid_mask = (delta_h_grid_q != 0.) & (delta_h_grid_m != 0.)
        h_grid_q_diff = delta_h_grid_m / delta_h_grid_q
        h_grid_q_diff[grid_mask==False] = 1

        self.pts_cleaned = Sector_bev.transfer_features_to_points(sector_coors, h_grid_q_diff)

        self.valid_mask = (self.N_r_M != -1) & (self.N_s_M != -1) & \
                     (self.histo_pts > self.cfg['minimum_nbr_pts']) & (self.histo_pts_q > self.cfg['minimum_nbr_pts']) #& \
                     # (self.pts_cleaned > self.cfg['scan_ratio_test_threshold'])# & \
                     # (self.min_h_M_pts > 0.2) #(min_h_q_pts > 0.2) &\


        dyn_props[self.valid_mask==False] = 1

        # visualize_points3D(self.voi_points_m, dyn_props)
        # visualize_points3D(self.voi_points_m, dyn_props < 0.2)
        scan_ratio_M = dyn_props

        # scan_ratio_M = 2 * np.ones(sector_coors.shape[0])
        # Per-point scan ratio, unvalid and vacant bins are "static"
        # scan_ratio_M[self.valid_mask] = Sector_bev.transfer_features_to_points(sector_coors[self.valid_mask], scan_ratio_grid)

        return scan_ratio_M

    def fit_ground_plane(self, points):

        if len(points) < self.cfg['minimum_nbr_pts']:
            return np.zeros(points.shape[0])

        pca_model = PCA(n_components=3)
        z_mean = points[:, 2].mean()
        # init mask
        ground_bin_mask = points[:, 2] < z_mean + self.cfg['tau_seed_height']

        for _ in range(self.cfg['ground_plane_iterations']):
            if np.sum(ground_bin_mask) <= 2:
                ground_bin_mask = np.zeros(points.shape[0], dtype=bool)
                break

            pca_model.fit(points[ground_bin_mask, :3])
            col_idx = np.argmin(pca_model.explained_variance_)

            n_vector = pca_model.components_[:,col_idx]
            d_mean = - n_vector.T @ points[ground_bin_mask, :3].mean(0)
            d_dash = - n_vector.T @ points[ground_bin_mask, :3].T
            ground_bin_mask[ground_bin_mask] = d_mean - d_dash < self.cfg['tau_ground_margin']

        ground_bin_mask = ground_bin_mask

        # Eliminate points below the plane (z coordinate is smaller than plane point at cca same xy)
        # Addition from Patrik Vacek
        Bev = BEV((0.2,0.2))
        Bev.create_bev_template_from_points(points)
        height_grid = Bev.generate_bev(points[ground_bin_mask], points[ground_bin_mask][:, 2])
        new_height = Bev.transfer_features_to_points(points, height_grid)
        ground_bin_mask[new_height >= points[:,2]] = True


        return ground_bin_mask

    def ground_plane_fitting(self):
        self.ground_mask = np.zeros(self.dynamic_mask.shape[0], dtype=bool)

        for i in np.unique(self.N_r_M[self.dynamic_mask]):
            if i == -1: continue

            r_mask = self.N_r_M == i

            for j in np.unique(self.N_s_M[self.dynamic_mask & r_mask]):
                if j == -1: continue

                point_mask = (r_mask) & (self.N_s_M == j)
                bin_ground_mask = self.fit_ground_plane(self.voi_points_m[point_mask])
                self.ground_mask[point_mask] = bin_ground_mask

    @timeit
    def run_one_frame(self, frame_id):
        ''' all point variables and attributes are stored in class for visualization, storing and debugging '''
        # for frame_id in range(0,len(self.pcls)):
        print(f"Processing Frame: {frame_id:06d}")
        self.prepare_frame_data(frame_id)

        # Calculate bins and delta H for global MAP
        self.N_r_M, self.N_s_M = self.transfer_sectors_to_points(self.voi_points_m)
        self.delta_h_grid_m = self.calculate_heights_sector(self.voi_points_m, self.N_r_M, self.N_s_M)

        # Calculate bins and delta H for queue
        self.N_r_q, self.N_s_q = self.transfer_sectors_to_points(self.points_q)
        self.delta_h_grid_q = self.calculate_heights_sector(self.points_q, self.N_r_q, self.N_s_q)





        self.scan_ratio_M = self.calculate_scan_ratio(self.delta_h_grid_m, self.delta_h_grid_q)




        valid_mask = (self.N_r_M != -1) & (self.N_s_M != -1)

        self.dynamic_mask = np.zeros(self.scan_ratio_M.shape[0], dtype=bool)
        self.dynamic_mask[valid_mask] = self.scan_ratio_M[valid_mask] < self.cfg['scan_ratio_test_threshold']


        self.ground_plane_fitting()

        self.dynamic_mask[self.ground_mask] = 0

        # dynamic_points = self.all_dynamic[self.map_valid_mask].copy()
        # dynamic_points[self.dynamic_mask==True] = True

        self.all_ground[self.map_valid_mask] = self.ground_mask
        self.all_dynamic[self.map_valid_mask] = self.dynamic_mask



    def run_whole_sequence(self):
        for frame_id in range(len(self.pcls)):
            self.run_one_frame(frame_id)

        for frame_id in range(len(self.pcls)):
            if frame_id == 0:
                self.ground_label = self.all_ground[:self.pcl_size[frame_id]]
                self.dyn_label = self.all_dynamic[:self.pcl_size[frame_id]]
            else:
                self.ground_label = self.all_ground[self.pcl_size[frame_id - 1]: self.pcl_size[frame_id]]
                self.dyn_label = self.all_dynamic[self.pcl_size[frame_id - 1]: self.pcl_size[frame_id]]

            np.save(self.output_dir + f'/ground/{frame_id:06d}.npy', self.ground_label)
            np.save(self.output_dir + f'/dynamic/{frame_id:06d}.npy', self.dyn_label)

        # np.save(self.output_dir + f'/all_ground.npy', self.all_ground)
        # np.save(self.output_dir + f'/all_dynamic.npy', self.all_dynamic)






if __name__ == '__main__':

    import sys

    # sequence = int(sys.argv[1])
    sequence = 6
    dataset = Lidar_Dataset('semantic_kitti')

    # data = dataset.prep_dataset.get_data_for_erasor(sequence=sequence)# , start=2470, end=2500)
    # data = dataset.prep_dataset.get_data_for_erasor(sequence=sequence)
    data = dataset.prep_dataset.get_base_data(sequence=sequence, start=40, end=75)


    # pid = os.getpid()
    pid = 0
    output_dir = data['output_dir'] + f"erasor_{pid}"
    pcls = data['pcls']
    poses = data['poses']
    dynamic_labels_data = data['dynamic_labels']
    pcl_size = data['pcls_point_range']




    Er = Erasor(output_dir)

    Er.preload_data(pcls,poses,pcl_size)
    # Er.run_one_frame(10)
    # Er.run_one_frame(12)
    # Er.run_one_frame(13)

    Er.run_whole_sequence()




    # Er.run_one_frame(0)

    # Er.run_one_frame(2)
    # Er.run_one_frame(3)


    # visualize_points3D(Er.points_m, Er.all_dynamic)

    p = np.concatenate((Er.vis_points_q, Er.points_m))
    l = np.concatenate((np.ones(Er.points_q.shape[0]), Er.all_dynamic * 2))

    visualize_points3D(p, l)
