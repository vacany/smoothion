import numpy as np
from tqdm import tqdm

# from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence
# from my_datasets.structures.bev import BEV
# from my_datasets.visualizer import *

def visibility_freespace(curr_pts, pose, cfg):
    '''
    Local point cloud and lidar position with respect to the point local frame. Then it is consistent.
    :param curr_pts: point cloud for raycasting
    :param pose: pose of lidar from where the beams are raycasted
    :param cfg: config file with cell sizes etc.
    :return: point cloud of safely visible areas based on lidar rays
    '''

    cell_size = cfg['cell_size']
    size_of_block = cfg['size_of_block']
    # Sort the point from closest to farthest
    distance = np.sqrt(curr_pts[..., 0] ** 2 + curr_pts[..., 1] ** 2 + curr_pts[..., 2] ** 2)
    index_by_distance = distance.argsort()
    curr_pts = curr_pts[index_by_distance]

    # Get the boundaries of the raycasted point cloud
    x_min, x_max, y_min, y_max, z_min, z_max = cfg['x_min'], cfg['x_max'], cfg['y_min'], cfg['y_max'], cfg['z_min'], cfg['z_max']

    x_min -= 2
    y_min -= 2
    z_min -= 1

    x_max += 2
    y_max += 2
    z_max += 1

    # Create voxel grid
    xyz_shape = np.array(
            (np.round((x_max - x_min) / cell_size[0]) + 3,
             np.round((y_max - y_min) / cell_size[1]) + 3,
             np.round((z_max - z_min) / cell_size[2]) + 3),
            dtype=int)

    # 0 is no stat, -1 is block, 1 is free, 2 is point
    cur_xyz_voxel = np.zeros(xyz_shape)
    accum_xyz_voxel = np.zeros(xyz_shape)

    curr_xyz_points = np.array(np.round(((curr_pts[:, :3] - np.array((x_min, y_min, z_min))) / cell_size)), dtype=int)
    cur_xyz_voxel[curr_xyz_points[:, 0], curr_xyz_points[:, 1], curr_xyz_points[:, 2]] = 2

    # Iterate one-by-one and update the voxel grid with visibility and blockage for next rays
    for p in curr_pts:
        # Calculate number of intermediate points based on the cell size of voxel grid
        nbr_inter = int(cfg['x_max'] / cell_size[0])
        # Raycast the beam from pose to the point
        ray = np.array((np.linspace(pose[0], p[0], nbr_inter),
                       np.linspace(pose[1], p[1], nbr_inter),
                          np.linspace(pose[2], p[2], nbr_inter))).T

        # Transform the ray to voxel grid coordinates
        xyz_points = np.array(np.round(((ray[:, :3] - np.array((x_min, y_min, z_min))) / cell_size)), dtype=int)
        # xyz_points = xyz_points[((xyz_points != xyz_points[-1]).all(1)) & ((xyz_points != xyz_points[0]).all(1))] # leave last and first cell
        # xyz_points = xyz_points[(xyz_points[:,2] != xyz_points[-1,2] - 1) &
        #                         (xyz_points[:,2] != xyz_points[-1,2]) &
        #                         (xyz_points[:,2] != xyz_points[-1,2] + 1)]

        # find the intersection of ray and current status of voxels
        ray_stats = cur_xyz_voxel[xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]]

        if len(xyz_points) == 0: continue  # if the ray is eliminated by security checks

        # Take last point of the ray and create blockage around it for other rays (to create occlusion)
        last_ray_pts = xyz_points[-1]
        cur_xyz_voxel[last_ray_pts[0] - (size_of_block + 1): last_ray_pts[0] + size_of_block,
        last_ray_pts[1] - (size_of_block + 1): last_ray_pts[1] + size_of_block,
        last_ray_pts[2] - (size_of_block + 1): last_ray_pts[2] + size_of_block] = - 1

        # Take only the part of ray before the blockage
        if (ray_stats == -1).any():
            # find the first intersection index
            first_intersection = (np.where(ray_stats == -1)[0][0])
            xyz_points = xyz_points[:first_intersection]

        # Update voxel grid with the visibility of the ray
        cur_xyz_voxel[xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]] = 1
        accum_xyz_voxel[xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]] += 1

    # point_coords = np.argwhere(cur_xyz_voxel == 2)
    ray_coords = np.argwhere(cur_xyz_voxel == 1)
    # blocks_coords = np.argwhere(cur_xyz_voxel == -1)

    accum_freespace_feature = accum_xyz_voxel[ray_coords[:,0], ray_coords[:,1], ray_coords[:,2]]

    # restore the original coordinates in meters and add x,y,z, freespace feature
    accum_freespace_meters = ray_coords[:, :3] * cell_size + np.array((x_min, y_min, z_min))
    accum_freespace_meters = np.insert(accum_freespace_meters, 3, accum_freespace_feature, axis=1)

    return accum_freespace_meters


def accumulate_static_points(global_pts_list: list, cell_size=(0.2, 0.2, 0.2)):
    '''

    :param global_pts_list: List of pts in global coordinate system, len(global_pts_list) >= required_static_times
    :param mask_of_interest: Mask for list where to find static points, len(mask_of_interest) == len(global_pts_list)
    :param required_static_times: Nbr of times a point is accumulated in voxel to be considered static
    :param cell_size: voxel size
    :return: List of static points in the same format as global_pts_list, the number of occurence of global voxel grid per time
    '''
    # len pts bude 30, required prostrednich 10, a prepisovat?

    # calculate max and min coordinates of global pcl list
    global_pcl = np.concatenate(global_pts_list)
    # x_min, x_max = np.min(global_pcl[:, 0]), np.max(global_pcl[:, 0])
    # y_min, y_max = np.min(global_pcl[:, 1]), np.max(global_pcl[:, 1])
    # z_min, z_max = np.min(global_pcl[:, 2]), np.max(global_pcl[:, 2])

    static_mask_list = []
    for pts in global_pts_list:
        static_mask_list = compare_points_to_static_scene(global_pcl, pts, cell_size)

    # create voxel grid for points in global_pcl that min and max coordinates are boundaries of the grid
    # accum_grid = np.zeros((int((x_max - x_min) / cell_size[0] + 1), int((y_max - y_min) / cell_size[1] + 1),
    #                        int((z_max - z_min) / cell_size[2] + 1)))


    # for global_pts in global_pts_list:
    #     # Accumulate all points in voxel grid
    #     shifted_global_pts = global_pts[:, :3] - np.array([x_min, y_min, z_min])
    #     shifted_coors = shifted_global_pts / cell_size
    #
    #     grid = np.zeros(accum_grid.shape)
    #
    #     grid[shifted_coors[:, 0].astype(int), shifted_coors[:, 1].astype(int), shifted_coors[:, 2].astype(int)] = 1
    #     accum_grid += grid
    #
    # static_mask_list = []
    #
    # for global_pts in global_pts_list:
    #     # assign each point in global_pts to a voxel and get the static mask
    #     shifted_global_pts = global_pts[:, :3] - np.array([x_min, y_min, z_min])
    #     shifted_coors = shifted_global_pts / cell_size
    #     static_mask = accum_grid[
    #         shifted_coors[:, 0].astype(int), shifted_coors[:, 1].astype(int), shifted_coors[:, 2].astype(int)]
    #
        # static_mask_list.append(static_mask)

    return static_mask_list


def compare_points_to_static_scene(global_pcl_list, points, cell_size):
    '''

    :param pcls: list of helper point to decide if variable "points" is static or dynamic
    :param points: Point cloud of interest
    :param cell_size:
    :return:
    '''

    pcls = np.concatenate(global_pcl_list)

    Bev = BEV(cell_size=(cell_size[0], cell_size[1]))
    Bev.create_bev_template_from_points(*[pcls, points])
    cell_z = cell_size[2]

    z_iter = np.round((pcls[:, 2].max() - pcls[:, 2].min()) / cell_z)
    z_min = pcls[:,2].min()
    inside_mask = np.zeros(points.shape[0])

    for z_idx in range(int(z_iter)):
        z_range_mask_points = (points[:, 2] > (z_min + z_idx * cell_z)) &\
                              (points[:, 2] < (z_min + (z_idx + 1) * cell_z))

        accum_grid = np.zeros(Bev.grid.shape)

        for global_pts in global_pcl_list:
            z_range_mask_pcls = (global_pts[:, 2] > (z_min + z_idx * cell_z)) & \
                                (global_pts[:, 2] < (z_min + (z_idx + 1) * cell_z))

            accum_grid += Bev.generate_bev(global_pts[z_range_mask_pcls], features=1)
            # print(print(global_pts[z_range_mask_pcls].shape), np.unique(accum_grid))

        inside_mask[z_range_mask_points] = Bev.transfer_features_to_points(points[z_range_mask_points], accum_grid)
        # print(z_idx)
    # print('inside_mask', np.unique(inside_mask))
    return inside_mask

def transfer_voxel_visibility(accum_freespace : np.ndarray, global_pts, cell_size):
    '''

    :param accum_freespace: Accumulated freespace feature
    :param global_pts: Point cloud of interest
    :param cell_size: Voxel size
    :return: Mask of points that are visible from the lidar
    '''
    # calculate max and min coordinates of global pcl list
    # acc_x_min, acc_x_max = np.min(accum_freespace[:, 0]), np.max(accum_freespace[:, 0])
    # acc_y_min, acc_y_max = np.min(accum_freespace[:, 1]), np.max(accum_freespace[:, 1])
    # acc_z_min, acc_z_max = np.min(accum_freespace[:, 2]), np.max(accum_freespace[:, 2])

    x_min, x_max = np.min(global_pts[:, 0]), np.max(global_pts[:, 0])
    y_min, y_max = np.min(global_pts[:, 1]), np.max(global_pts[:, 1])
    z_min, z_max = np.min(global_pts[:, 2]), np.max(global_pts[:, 2])

    # x_min, x_max = np.min([x_min, acc_x_min]), np.max([x_max, acc_x_max])
    # y_min, y_max = np.min([y_min, acc_y_min]), np.max([y_max, acc_y_max])
    # z_min, z_max = np.min([z_min, acc_z_min]), np.max([z_max, acc_z_max])


    filtered_accum = accum_freespace[(accum_freespace[:,0] > x_min) & (accum_freespace[:,0] < x_max) & \
                                        (accum_freespace[:,1] > y_min) & (accum_freespace[:,1] < y_max) & \
                                        (accum_freespace[:,2] > z_min) & (accum_freespace[:,2] < z_max)]


    voxel_grid = np.zeros((int((x_max - x_min) / cell_size[0] + 2), int((y_max - y_min) / cell_size[1] + 2),
                           int((z_max - z_min) / cell_size[2] + 2)))

    # not indices, but coordinates...
    global_pts_idx = np.round((global_pts[:, :3] - np.array([x_min, y_min, z_min])) / cell_size).astype(int)
    filtered_accum_idx = np.round((filtered_accum[:, :3] - np.array([x_min, y_min, z_min])) / cell_size).astype(int)

    voxel_grid[filtered_accum_idx[:,0].astype(int), filtered_accum_idx[:,1].astype(int), filtered_accum_idx[:,2].astype(int)] += 1

    static_mask = voxel_grid[global_pts_idx[:,0], global_pts_idx[:,1], global_pts_idx[:,2]]

    return static_mask

def closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=False, clampA0=False, clampA1=False, clampB0=False,
                                clampB1=False):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)


            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)




if __name__ == '__main__':


    import tqdm
    from sklearn.neighbors import NearestNeighbors

    import torch

    n = 32000
    K = 1

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for i in tqdm.tqdm(range(K * n)):
        ray = torch.rand(300,3, device=device)
        smooth_ray = torch.rand(320, 3, device=device)

        torch.cdist(ray, smooth_ray).min()
        # dist = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(ray).kneighbors(smooth_ray, return_distance=True)[0]

        # if dist.min() > 0.1:
        #     pass    # denotes overlap
