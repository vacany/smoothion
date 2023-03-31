import os.path
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from my_datasets.structures.bev import BEV
from motion_supervision.generate_priors import correct_the_dynamic_priors



def ego_points(ego_box, cell_size=(0.2,0.2)):
    '''

    :return: point cloud of ego
    '''
    x, y, z = ego_box['translation']
    l, w, h = ego_box['size']
    yaw = Rotation.from_matrix(ego_box['rotation']).as_rotvec()[2]

    safe_margin = 0.5   # for points inside all cells. Is always same

    pcl_list = []
    for i in range(int(l / cell_size[0] / safe_margin)):
        for j in range(int(w / cell_size[1] / safe_margin)):
            pcl_list.append(
                np.array((i * cell_size[0] * safe_margin, j * cell_size[1] * safe_margin, z), dtype=float))

    pcl = np.stack(pcl_list)

    xy_shift = np.array((l / 2, w / 2, 0), dtype=float)

    pcl -= xy_shift

    return pcl

def get_ego_pts_in_all_poses(ego_pts, ego_poses):
    '''
    :param ego_pts: point cloud of ego
    :param ego_poses: poses of ego
    :return: point cloud of ego in all poses
    '''
    ego_pts_in_all_poses = []

    for t, pose in enumerate(ego_poses):

        one_ego_pts = np.insert(ego_pts, 3, 1, axis=1)
        one_ego_pts = one_ego_pts @ pose.T

        one_ego_pts = np.insert(one_ego_pts[:, :3], 3, t, axis=1)
        ego_pts_in_all_poses.append(one_ego_pts[:, :4])

    return np.concatenate(ego_pts_in_all_poses)

def get_driven_pts(global_pts, all_ego_pts, ego_box, cell_size=(0.2,0.2)):

    bev = BEV(cell_size=(cell_size[0], cell_size[1]))
    bev.create_bev_template_from_points(*[all_ego_pts, global_pts])

    # -1 is reserved for not driven
    time_grid = bev.generate_bev(all_ego_pts, all_ego_pts[:, 3], init_value=-1)
    height_grid = bev.generate_bev(all_ego_pts, all_ego_pts[:, 2], init_value=-1)

    ego_driven = bev.transfer_features_to_points(global_pts, time_grid)
    ego_driven_height = bev.transfer_features_to_points(global_pts, height_grid)

    above_limit = global_pts[:, 2] > ego_driven_height + ego_box['size'][2] / 2 # this can be done better maybe, idk

    ego_driven[above_limit] = -1

    return ego_driven

def variance_based_object_proposals(global_pts, ego_driven, z_var_object=0.5, z_var_road=0.1, min_road_pts=10, z_var_outlier=4):
    '''

    :param global_pts: global point cloud
    :param ego_driven: time mask when have ego driven
    :param z_var_object: criterion to decide, if there is object on the place or not
    :return: mask of object proposals in whole box (including the road points as true) for further processing
    '''
    times = np.unique(ego_driven[ego_driven != -1])

    object_proposals = np.zeros(ego_driven.shape, bool)
    road_proposals = np.zeros(ego_driven.shape, bool)

    for t in times:

        proposal_pts = global_pts[ego_driven == t]
        z_var = proposal_pts[:, 2].max() - proposal_pts[:, 2].min()
        if z_var > z_var_object and z_var < z_var_outlier:
            object_proposals[ego_driven == t] = True

        if z_var < z_var_road and proposal_pts.shape[0] > min_road_pts:
            road_proposals[ego_driven == t] = True

    # remove points above ego bounding box by pose - in ego driven already
    # tohle to uplne neresi, kvuli tomu jak je napsana funkce podtim. Vyzaduje to vic uprav.
    # above_ego_mask = global_pts[:, 2] > pose[2, -1] # pose is here the lidar
    # object_proposals[above_ego_mask] = False

    return object_proposals, road_proposals

def split_proposals_to_object_and_road(global_pts, ego_driven, object_proposals, cell_size=(0.2,0.2), z_var_proposal=0.8, Z_VAR_ROAD=0.1, MIN_ROAD_PTS=10):
    '''

    :param global_pts: global point cloud
    :param object_proposals: mask of object proposals in whole box (including the road points as true)
    :param z_var_road: criterion to decide, if there is object on the place or not
    :return: mask of object proposals in whole box (including the road points as true) for further processing
    '''

    global_object_mask = np.zeros(ego_driven.shape, bool)
    global_road_mask = np.zeros(ego_driven.shape, bool)


    for proposals_times in np.unique(ego_driven[object_proposals]):

        prop_pts = global_pts[ego_driven == proposals_times]


        bev = BEV(cell_size=(cell_size[0], cell_size[1]))
        bev.create_bev_template_from_points(prop_pts)
        # xy = bev.return_point_coordinates(prop_pts)

        z_var = prop_pts[:, 2].max() - prop_pts[:, 2].min()

        z_var_half = prop_pts[:, 2].max() - 0.5 * z_var
        # z_var_low = prop_pts[:, 2].min() + 0.1 * z_var

        above_pts = prop_pts[prop_pts[:, 2] > z_var_half]
        grid = bev.generate_bev(above_pts, 1, init_value=0)
        area_mask = bev.transfer_features_to_points(prop_pts, grid).astype(bool)

        # Jinak je v semantickitti nahla rovina cesty a tam se to vysere

        # above boundary
        area_var = prop_pts[area_mask, 2].max() - prop_pts[area_mask, 2].min()
        below_boundary = prop_pts[area_mask, 2].min() + Z_VAR_ROAD * area_var  # 0.1 is safe parameter? for each point in cell separately?
        upper_boundary = prop_pts[area_mask, 2].max() + Z_VAR_ROAD * area_var

        road_mask = prop_pts[area_mask, 2] < below_boundary
        object_mask = (prop_pts[area_mask, 2] > below_boundary) & \
                      (prop_pts[area_mask, 2] < upper_boundary)


        assign_idx = np.argwhere(ego_driven == proposals_times)[area_mask, 0]

        # Tenhle threshold je kvuli pripadu z dalky, kde je videt malo bodu a muze se priradit spatne
        if np.sum(road_mask) > MIN_ROAD_PTS:
            global_road_mask[assign_idx[road_mask]] = True

        # double check if rest of the points are still inside the z-var range
        if len(assign_idx[object_mask]) > 0:
            # print('assigned!')
            object_prop_variance = global_pts[assign_idx[object_mask], 2].max() - global_pts[assign_idx[object_mask], 2].min()
            if object_prop_variance > z_var_proposal:
                global_object_mask[assign_idx[object_mask]] = True

    return global_object_mask, global_road_mask

def split_objects_based_on_adjacent_cellsize(global_pts, object_mask, cell_size=(0.2,0.2)):
    '''
    This mimic the contours of proposals based on grid. It just connects all neighbours in grid and assign id.
    This is not perfect instance segmentation. They are proposals for masked learning of instance seg.
    Clustering is based on x-y coordinates only.
    :param global_pts:
    :param object_mask:
    :return:
    '''
    orig_cluster_mask = - np.ones(object_mask.shape, int)
    indices = np.argwhere(object_mask)[:,0]

    clusters = DBSCAN(eps=np.min(cell_size), min_samples=3).fit_predict(global_pts[object_mask][:, :2])

    orig_cluster_mask[indices] = clusters

    return orig_cluster_mask

def run_ego_prior(global_pts, all_ego_pts, ego_box, cfg, cluster=False ):
    # todo assign config file

    object_ids = - np.ones(global_pts.shape[0])

    ego_driven = get_driven_pts(global_pts, all_ego_pts, ego_box, cell_size=cfg['cell_size'])

    # can be done in whole pts i guess - cannot because of object overlaps. But should kind of works anyways.
    # maybe not a bad idea to do it in whole pts, but probably will be much noise and need to have safer margin

    object_proposals, road_proposals = variance_based_object_proposals(global_pts, ego_driven,
                                                                        z_var_object=cfg['z_var_proposal'],
                                                                        z_var_road=cfg['z_var_road'],
                                                                        min_road_pts=cfg['min_road_pts'],
                                                                        z_var_outlier=cfg['z_var_outlier'])

    object_mask, road_mask = split_proposals_to_object_and_road(global_pts, ego_driven, object_proposals, cell_size=cfg['cell_size'], z_var_proposal=cfg['z_var_proposal'], Z_VAR_ROAD=cfg['z_var_road'],
                                                                MIN_ROAD_PTS=cfg['min_road_pts'])
    # road mask is not used at this point
    if cluster:
        object_ids = split_objects_based_on_adjacent_cellsize(global_pts, object_mask, cell_size=cfg['cell_size'])

    return object_mask, road_proposals, object_ids

# todo def generate_labels_for_mask_instance_segmentation
# todo kitti has this bad reflections, Do not know if it is issue

# todo remove other points and focus on radius around the point
def plot_point_cloud_area(vis_pcl, features, save=None, vmin=0, vmax=1, cb=True):

    pcl = vis_pcl.copy()
    pcl -= vis_pcl.mean(axis=0)

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(projection='3d')

    xs = pcl[:, 0]
    ys = pcl[:, 1]
    zs = pcl[:, 2]


    colorbar_data = ax.scatter(xs, ys, zs, marker='.', s=0.3, c=features, alpha=0.6, vmin=vmin, vmax=vmax, cmap='jet')
    ax.set_xlim([- pcl[:,0].min() - 1, pcl[:,0].max() + 1])
    ax.set_ylim([- pcl[:,1].min() - 1, pcl[:,1].max() + 1])
    ax.set_zlim([- pcl[:,2].min() - 1, pcl[:,2].max() + 1])

    ax.view_init(elev=25, azim=210)
    ax.dist = 50

    if cb:
        fig.colorbar(colorbar_data)

    # ax.set_aspect('equal')
    ax.set_box_aspect((1, 1, 1))
    plt.axis('off')

    if save is not None:
        try:
            os.makedirs(os.path.dirname(save), exist_ok=True)
        except:
            pass

        plt.savefig(save)
    else:
        plt.show()

    plt.close()

def plot_FP(global_pts, ego_prior_label, dynamic_label, wrong_indices, save_path):
    # USE CASE saving visuals
    valid_mask = ego_prior_label != -1
    valid_mask_idx = np.argwhere(valid_mask)[:, 0]
    wrong_priors = np.argwhere(dynamic_label[valid_mask] != ego_prior_label[valid_mask])[:, 0]

    if len(wrong_priors) > 0:
    #     save_path = f'{os.path.expanduser("~")}/data/visuals/{sequence.sequence}_{frame:06d}.png'
    #     plot_FP(global_pts, ego_prior_label, wrong_priors, save_path)

        wrong_orig_idx = valid_mask_idx[wrong_indices]

        randomly_sampled_idx = np.random.choice(wrong_orig_idx)
        dist_around_sampled = np.linalg.norm(global_pts - global_pts[randomly_sampled_idx], axis=1) < 8

        # add iou diff as well
        plot_point_cloud_area(global_pts[dist_around_sampled], ego_prior_label[dist_around_sampled], vmin=-1, vmax=1,
                          cb=False, save=save_path)




def generate_ego_priors_for_sequence(sequence, cfg):
    '''
    Ego prior : -1 unknown, 0 static, 1 dynamic
    :param sequence:
    :param cfg:
    :return:
    '''
    poses = sequence.get_ego_poses()

    ego_pts = ego_points(sequence.ego_box)
    all_ego_pts = get_ego_pts_in_all_poses(ego_pts, poses)


    for frame in tqdm(range(len(sequence))):
        global_pts = sequence.get_global_pts(frame, 'lidar')

        object_mask, road_proposal, object_id = run_ego_prior(global_pts, all_ego_pts, sequence.ego_box,
                                                          cfg, cluster=False)

        ego_prior_label = - np.ones(global_pts.shape[0], int)
        ego_prior_label[object_mask] = 1


        local_pts = sequence.get_feature(frame, 'lidar')
        invalid_pts = np.linalg.norm(local_pts[:, :3], axis=1) < sequence.min_sensor_radius

        ego_prior_label[invalid_pts] = -1

        # label_file = sequence.sequence_path + '/prior_mos_labels/' + str(frame).zfill(6) + '.label'
        # store_mos_format(ego_prior_label, label_file)

        sequence.store_feature(ego_prior_label, frame, name='ego_prior_label')
        sequence.store_feature(road_proposal, frame, name='road_proposal')

    # cfg['static_cell_size'] = np.array((cfg['cell_size'][0], cfg['cell_size'][1], cfg['cell_size'][0])) # just add same value
    # correct_the_dynamic_priors(sequence, cfg=cfg) # this can be merged later

if __name__ == '__main__':
    from motion_supervision.constants import ego_prior_params
    from my_datasets.waymo.waymo import Waymo_Sequence
    from my_datasets.kitti.semantic_kitti import SemanticKitti_Sequence
    # from my_datasets.visualizer import *

    if len(sys.argv) > 1:
        seq = int(sys.argv[1])
    else:
        seq = 4

    sequence = SemanticKitti_Sequence(seq)

    generate_ego_priors_for_sequence(sequence, ego_prior_params)



    # Getting pseudo ids by taking nearby cells (that is as the voxel representation). Implemented by DBSCAN with fixed epsilon.
            # can I do it in postprocessing?

# visualize_points3D(global_pts, object_ids)
# map, this will lead to building the hd map as well
