import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from FastFlow3D.utils.pillars import remove_out_of_bounds_points
from motion_supervision.visibility import visibility_freespace, accumulate_static_points, compare_points_to_static_scene, transfer_voxel_visibility
from my_datasets.structures.bev import BEV
from timespace import box_utils

# from Neural_Scene_Flow_Prior.my_flow import Small_Scale_SceneFlow_Solver

from my_datasets.visualizer import *

# from timespace.box_utils import get_bbox_points

def store_mos_format(prior_label, save_path):
    prior_label_format = np.zeros(prior_label.shape, dtype=np.uint32)
    prior_label_format[prior_label == 1] = 251  # dynamic
    prior_label_format[prior_label == 0] = 9    # static
    prior_label_format[prior_label == -1] = 0   # unlabelled

    upper_half = prior_label_format >> 16  # get upper half for instances
    lower_half = prior_label_format & 0xFFFF  # get lower half for semantics
    # lower_half = remap_lut[lower_half]  # do the remapping of semantics
    label = (upper_half << 16) + lower_half  # reconstruct full label
    label = label.astype(np.uint32)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    label.tofile(save_path)

def select_boxes_in_pts_range(pts, ego_boxes):
    '''
    Select boxes that are in range of the points and take only the first consecutive boxes. This is done to avoid
    large SemanticKITTI and KITTI sequneces, where there is a loop closure with probably wrong SLAM poses.
    :param pts:
    :param ego_boxes:
    :return: Filtered ego boxes
    '''

    ego_positions = np.stack([ego_boxes[i]['translation'] for i in range(len(ego_boxes))])  # and time "consecutivnes"
    ego_positions = np.insert(ego_positions, 3, np.arange(0, len(ego_positions)), axis=1)

    # get points x_max, min and y_max, min
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])

    # take only ego positions within the range x_min, x_max, y_min, y_max
    position_mask  = (ego_positions[:, 0] > x_min) & (ego_positions[:, 0] < x_max) & \
                     (ego_positions[:, 1] > y_min) & (ego_positions[:, 1] < y_max)

    # filter positions out of pts range
    ego_box_sequence = ego_boxes[position_mask].copy()
    # check if time is consecutive in ego_positions
    consecutive_time = np.diff(ego_positions[position_mask, 3]) == 1
    # take first sequence of truth values from consecutive_time
    if not np.all(consecutive_time):
        first_false = np.argmin(consecutive_time)  # when multiple false, the argmin will return the first one
        ego_box_sequence = ego_box_sequence[:first_false].copy()
    # else:
    #     ego_box_sequence = ego_boxes.copy()

    return ego_box_sequence

# you must document the versions of algorithm!!!
# vyser se na ten case, kdy body objektu maji jen jednu lajnu bodu - uvest v limitations -
# To by ale vetsinou nemel byt ani problem, protoze ego prijede bliz a vidi vic.

def mark_by_driving_ver2(pts, boxes, z_var_object, z_var_road):
    pass

def mark_by_driving(pts, boxes, dyn_z_add=(1,1), stat_z_add=(10,0), z_object=0.3, z_road=0.1, return_time_mask=False):
    '''
    Produce mask what was driven through in different time
    :param pts:
    :param boxes:
    :param dyn_z_add:
    :param stat_z_add:
    :param return_time_mask:
    :return:
    '''
    marked_mask = - np.ones(pts.shape[0], dtype='int16')
    time_mask = - np.ones(pts.shape[0], dtype='int16')

    # add cohorent box number
    # eliminate boxes outside of the scene

    for box_id, ego_box in enumerate(boxes.copy()):

        # add lowe ratio to decide if points are road or objects
        inside_box = box_utils.get_point_mask(pts, ego_box, z_add=dyn_z_add)

        # marked_mask[inside_box] = 2
        # continue

        if inside_box.sum() < 3:
            continue

        variance_inside = pts[inside_box, 2].max() - pts[inside_box, 2].min()
        # print(box_id, variance_inside, np.sum(inside_box))

        if variance_inside > z_object:
            # dynamic object
            # marked_mask[inside_box] = 1
            time_mask[inside_box] = box_id

            # upper_half_points = inside_box & (pts[:, 2] >= ego_box['translation'][2]) & (pts[:, 2] < ego_box['translation'][2] + ego_box['size'][2] / 2)  # - ego_box[5] / 2)  # todo can be increased for security
            above_road_points = inside_box & (pts[:, 2] > pts[inside_box, 2].min() + z_road)
            marked_mask[above_road_points] = 1
        # road
        else:
            # if not np.any(upper_half_points):   # get additional when sky is the limit ... :D It will work, but static priors can deal with it
                # low_box_points = inside_box #& (pts[:,2] < ego_box[2])   # lower half and everything below the box
            below_box = box_utils.get_point_mask(pts, ego_box, z_add=stat_z_add)

            if below_box.sum() < 3:
                continue

            road_variance = pts[below_box, 2].max() - pts[below_box, 2].min()

            # lower_decile_points = below_box & (pts[:, 2] < ego_box['translation'][2] - ego_box['size'][2] / (5 / 10))  # lower decile
            # lower_decil_points = inside_box #& (pts[:, 2] < ego_box['translation'][2]) # - ego_box['size'][2] / (1 / 10)) #& (pts[:,2] < ego_box[2])   # lower half and everything below the box
            # print("ROAD: ", road_variance, np.sum(lower_decile_points))
            # marked_mask[lower_decile_points] = 0
            if road_variance < z_road:
                marked_mask[below_box] = 0

            # time_mask[upper_half_points] = box_id

    if return_time_mask:
        return marked_mask, time_mask
    else:
        return marked_mask


#
# def run_ego_priors(sequence, cfg):
#     ego_boxes = sequence.get_ego_boxes()
#
#     for frame in tqdm(range(len(sequence))):
#
#         if sequence.has_feature(frame, 'ego_label'):
#             continue
#         else:
#             local_pts = sequence.get_feature(frame, 'lidar')
#             sequence.store_feature(np.zeros(local_pts.shape[0]), frame, name='ego_label')
#
#         global_pts = sequence.get_global_pts(idx=frame, name='lidar')
#         local_pts = sequence.get_feature(idx=frame, name='lidar')
#
#         ego_label_orig  = - np.ones(local_pts.shape[0], dtype='int16')
#
#         # remove ego points
#         sensor_valid_mask = np.linalg.norm(local_pts[:, :3], axis=1) > sequence.min_sensor_radius  #
#
#         _, config_mask = remove_out_of_bounds_points(local_pts, cfg['x_min'], cfg['x_max'], cfg['y_min'],
#                                                   cfg['y_max'],
#                                                   cfg['z_min'], cfg['z_max'])
#
#
#
#         # Take only the boxes that are in range of the points and not loop closure
#         ego_box_sequence = select_boxes_in_pts_range(global_pts[config_mask], ego_boxes)
#
#         # this should be fast already without mask
#         ego_label, tm = mark_by_driving(global_pts[config_mask], ego_box_sequence, return_time_mask=True)
#
#         ego_label_orig[config_mask] = ego_label
#         ego_label_orig[sensor_valid_mask==False] = -1
#         # tm[sensor_valid_mask==False] = -1
#
#         sequence.store_feature(ego_label_orig, idx=frame, name='ego_label')
#         # sequence.store_feature(tm, idx=frame, name='ego_tm')

#
# def run_flow_solver_on_ego(sequence, flow_solver=None):
#     if flow_solver is None:
#         Flow_solver = Small_Scale_SceneFlow_Solver()
#     else:
#         Flow_solver = flow_solver
#
#     for frame in tqdm(range(0, len(sequence) - 1)):
#
#
#         pts1 = sequence.get_feature(idx=frame, name='lidar')
#         ego_label1 = sequence.get_feature(idx=frame, name='ego_label')
#
#         # prior objects
#         object_mask1 = ego_label1 == 1
#         road_mask1 = ego_label1 == 0
#
#         if np.sum(object_mask1) < 3:
#             continue
#
#         ego_pts1 = pts1[object_mask1]
#
#         pts2 = sequence.get_feature(idx=frame + 1, name='lidar')
#         ego_label2 = sequence.get_feature(idx=frame + 1, name='ego_label')
#         object_mask2 = ego_label2 == 1
#         ego_pts2 = pts2[object_mask2]
#
#         ego_flow = Flow_solver.generate_sceneflow(ego_pts1, ego_pts2)
#
#         frame_flow = - np.ones([pts1.shape[0], 4])
#         frame_flow[object_mask1, :3] = ego_flow.detach().cpu().numpy()
#         frame_flow[object_mask1, 3] = 1
#
#         sequence.store_feature(frame_flow, idx=frame, name='ego_flow')


def process_freespace(param_list):
    sequence, cfg, i = param_list
    print("Freespace frame: ", i)
    local_pts = sequence.get_feature(i, 'lidar')

    # REMOVE ego points
    sensor_pose = sequence.sensor_position
    sensor_valid_mask = np.linalg.norm(local_pts[:, :3] - sensor_pose[:3], axis=1) > sequence.min_sensor_radius  #

    curr_pts, _ = remove_out_of_bounds_points(local_pts[sensor_valid_mask], cfg['x_min'], cfg['x_max'],
                                              cfg['y_min'],
                                              cfg['y_max'],
                                              cfg['z_min'], sensor_pose[2])

    accum_freespace_meters = visibility_freespace(curr_pts, sensor_pose, cfg)

    ######
    # -------------      Lidar  X  ----------------
    # freespace freespace freespace freespace freespace
    # -------------      Ground    ----------------
    ######

    below_ego_center_height = accum_freespace_meters[:, 2] > sensor_pose[2] - sequence.ego_box['size'][2] / 1.5
    above_sensor_height = accum_freespace_meters[:, 2] < sensor_pose[2]  # cfg['freespace_max_height']
    height_band = below_ego_center_height & above_sensor_height

    accum_freespace_meters = accum_freespace_meters[height_band]

    sequence.store_feature(accum_freespace_meters, idx=i, name='accum_freespace')

def generate_freespace_from_sequence(sequence, cfg):
    '''
    :param sequence: a sequence of point clouds
    :param cfg: configuration file
    :return: freespace features for each point cloud in the sequence stored inside the sequence folder
    '''
    param_list = [[sequence, cfg, i] for i in range(len(sequence))]

    pool = Pool()  # Create a multiprocessing Pool
    pool.map(process_freespace, param_list)

    pool.close()
    pool.join()


    # THIS IS A ORIGINAL
    # for i in tqdm(range(len(sequence))):
    #     if i == 5: break
        # if sequence.has_feature(idx=i, name='accum_freespace'):
        #     continue
        # else:
        #     local_pts = sequence.get_feature(i, 'lidar')
        #     sequence.store_feature(np.zeros(local_pts.shape[0]), i, name='accum_freespace')

        # local_pts = sequence.get_feature(i, 'lidar')
        #
        # # REMOVE ego points
        # sensor_pose = sequence.sensor_position
        # sensor_valid_mask = np.linalg.norm(local_pts[:, :3] - sensor_pose[:3], axis=1) > sequence.min_sensor_radius #
        #
        #
        # curr_pts, _ = remove_out_of_bounds_points(local_pts[sensor_valid_mask], cfg['x_min'], cfg['x_max'], cfg['y_min'],
        #                                           cfg['y_max'],
        #                                           cfg['z_min'], sensor_pose[2])
        #
        # accum_freespace_meters = visibility_freespace(curr_pts, sensor_pose, cfg)
        #
        # ######
        # # -------------      Lidar  X  ----------------
        # # freespace freespace freespace freespace freespace
        # # -------------      Ground    ----------------
        # ######
        #
        # below_ego_center_height = accum_freespace_meters[:, 2] > sensor_pose[2] - sequence.ego_box['size'][2] / 1.5
        # above_sensor_height = accum_freespace_meters[:, 2] < sensor_pose[2] #cfg['freespace_max_height']
        # height_band = below_ego_center_height & above_sensor_height
        #
        # accum_freespace_meters = accum_freespace_meters[height_band]
        #
        # sequence.store_feature(accum_freespace_meters, idx=i, name='accum_freespace')

def generate_visibility_prior(sequence, cfg):

    accum_freespace = []

    for i in range(len(sequence)):
        # if i == 40:break
        freespace = sequence.get_feature(i, 'accum_freespace')
        current_pose = sequence.get_feature(i, 'pose')

        global_freespace = sequence.pts_to_frame(freespace, current_pose)

        accum_freespace.append(global_freespace)

    accum_freespace = np.concatenate(accum_freespace)

    for i in tqdm(range(len(sequence))):
        # if i != 4481: continue
        global_pts = sequence.get_global_pts(i, 'lidar')
        local_pts = sequence.get_feature(i, 'lidar')
        visibility_prior = np.zeros(global_pts.shape[0], dtype='int64')

        _, mask = remove_out_of_bounds_points(local_pts, cfg['x_min'], cfg['x_max'], cfg['y_min'], cfg['y_max'],
                                              cfg['z_min'], cfg['z_max'])
        # accum_freespace = []

        # max_diff = cfg['correction_time']    # todo calculate it in config file
        # if i >= max_diff and i < len(sequence) - max_diff:
        #     for adjacent_i in range(i - max_diff, i + max_diff + 1):
        #         # if i == 40:break
        #
        #         freespace = sequence.get_feature(adjacent_i, 'accum_freespace')
        #         current_pose = sequence.get_feature(adjacent_i, 'pose')
        #
        #         global_freespace = sequence.pts_to_frame(freespace, current_pose)
        #
        #         accum_freespace.append(global_freespace)
        #
        #     accum_freespace = np.concatenate(accum_freespace)


            # todo check this shit
        pts_in_freespace = transfer_voxel_visibility(accum_freespace, global_pts[mask], cell_size=cfg['cell_size'])

        visibility_prior[mask] = pts_in_freespace

        sequence.store_feature(visibility_prior, idx=i, name='visibility_prior')

def process_static(param_list):
    sequence, cfg, frame = param_list

    print("Static features: ", frame)

    if frame + cfg['required_static_time'] >= len(sequence):
        diff_to_end = len(sequence) - frame

        before_indices = list(range(frame - (cfg['required_static_time'] - diff_to_end), frame))
        after_frame_indices = list(range(frame, len(sequence)))

        indices = before_indices + after_frame_indices

    else:
        indices = list(range(frame, frame + cfg['required_static_time']))

    global_pts_list = []

    # accumulate freespace
    accum_freespace = []
    # get required static before and after
    accum_indices = range(frame - cfg['required_static_time'], frame + cfg['required_static_time'] + 1)

    for i in accum_indices:

        if i >= 0 and i < len(sequence):

            freespace = sequence.get_feature(i, 'accum_freespace')
            current_pose = sequence.get_feature(i, 'pose')

            global_freespace = sequence.pts_to_frame(freespace, current_pose)
            accum_freespace.append(global_freespace)

    accum_freespace = np.concatenate(accum_freespace)

    for i in indices:
        # global_pts_list = [sequence.get_global_pts(idx=i, name='lidar') for i in indices]

        # curr_pts = sequence.get_feature(idx=frame, name='lidar')
        # smaller_pts, mask = remove_out_of_bounds_points(curr_pts, cfg['x_min'], cfg['x_max'], cfg['y_min'],
        # cfg['y_max'], cfg['z_min'], cfg['z_max'])

        global_pts = sequence.get_global_pts(idx=i, name='lidar')
        # global_pts = global_pts[mask]

        global_pts_list.append(global_pts)

    # Run the static generation for point clouds of interest
    current_pts = sequence.get_global_pts(frame, 'lidar')

    static_prior = np.zeros(current_pts.shape[0], dtype='int')

    in_freespace_mask = transfer_voxel_visibility(accum_freespace, current_pts, cfg['static_cell_size'])
    # print(indices)
    for around_pts in global_pts_list:
        one_static_mask = transfer_voxel_visibility(around_pts, current_pts, cfg['static_cell_size'])
        static_prior += (one_static_mask > 0).astype('int')

    # point previously visible and mistakes of noise in odometry
    # this is for cases like the truck in SemanticKitti 27 sequence
    static_prior[in_freespace_mask == 1] = -1

    sequence.store_feature(static_prior, idx=frame, name='prior_static_mask')


def generate_static_points_from_sequence(sequence, cfg):
    # todo ADD visibility condition. If visible previously, then remove static label from all freespace? done
    # todo project static from this visibility condition. (not all because of trees over streets etc)

    accum_freespace = []
    for frame in range(len(sequence)):
        freespace = sequence.get_feature(frame, 'accum_freespace')
        current_pose = sequence.get_feature(frame, 'pose')

        global_freespace = sequence.pts_to_frame(freespace, current_pose)

        accum_freespace.append(global_freespace)

    accum_freespace = np.concatenate(accum_freespace)

    # create one voxel map
    # compare the voxel map with pts to generate in freespace mask
    # compare the points in it

    # param_list = [[sequence, cfg, i] for i in range(len(sequence))]
    #
    # pool = Pool()  # Create a multiprocessing Pool
    # pool.map(process_static, param_list)
    #
    # pool.close()
    # pool.join()

    for frame in tqdm(range(len(sequence))):
        if frame + cfg['required_static_time'] >= len(sequence):
            diff_to_end = len(sequence) - frame

            before_indices = list(range(frame - (cfg['required_static_time'] - diff_to_end), frame))
            after_frame_indices = list(range(frame, len(sequence)))

            indices = before_indices + after_frame_indices

        else:
            indices = list(range(frame, frame + cfg['required_static_time']))

        global_pts_list = []

        # accumulate freespace
        # accum_freespace = []
        # get required static before and after
        # accum_indices = range(frame - cfg['required_static_time'], frame + cfg['required_static_time'] + 1)

        # for i in accum_indices:
        #
        #     if i >= 0 and i < len(sequence):
        #
        #         freespace = sequence.get_feature(i, 'accum_freespace')
        #         current_pose = sequence.get_feature(i, 'pose')
        #
        #         global_freespace = sequence.pts_to_frame(freespace, current_pose)
        #         accum_freespace.append(global_freespace)
        #
        # accum_freespace = np.concatenate(accum_freespace)

        for i in indices:
            # global_pts_list = [sequence.get_global_pts(idx=i, name='lidar') for i in indices]

            # curr_pts = sequence.get_feature(idx=frame, name='lidar')
            # smaller_pts, mask = remove_out_of_bounds_points(curr_pts, cfg['x_min'], cfg['x_max'], cfg['y_min'],
            # cfg['y_max'], cfg['z_min'], cfg['z_max'])

            global_pts = sequence.get_global_pts(idx=i, name='lidar')
            # global_pts = global_pts[mask]

            global_pts_list.append(global_pts)

        # Run the static generation for point clouds of interest
        current_pts = sequence.get_global_pts(frame, 'lidar')

        static_prior = np.zeros(current_pts.shape[0], dtype='int')

        in_freespace_mask = transfer_voxel_visibility(accum_freespace, current_pts, cfg['static_cell_size'])
        # print(indices)
        for around_pts in global_pts_list:
            one_static_mask = transfer_voxel_visibility(around_pts, current_pts, cfg['static_cell_size'])
            static_prior += (one_static_mask > 0).astype('int')

        # point previously visible and mistakes of noise in odometry
        # this is for cases like the truck in SemanticKitti 27 sequence
        static_prior[in_freespace_mask == 1] = -1

        sequence.store_feature(static_prior, idx=frame, name='prior_static_mask')



def correct_the_dynamic_priors(sequence, cfg=None):

    corr_time = cfg['correction_time']

    for frame in tqdm(range(0, len(sequence))): # tmp

        # if frame != 4481: continue

        pts1 = sequence.get_global_pts(idx=frame, name='lidar')
        ego_prior1 = sequence.get_feature(idx=frame, name='ego_prior_label')

        # take adjacent frames based on equation, clip the start and end of sequence
        # The clipping results in smooth code and remove dynamic labels on first and last all together (pts1 = pts2 | pts3)
        adjacent_indices = [frame - corr_time, frame + corr_time]
        adjacent_indices = np.clip(adjacent_indices, a_min= 0, a_max= len(sequence) - 1)

        # preload chosen frames to set overlapping points from dynamic to "unlabelled"
        adjacent_frames = [sequence.get_global_pts(correction_frame, name='lidar') for correction_frame in adjacent_indices]
        adjacent_frames = np.concatenate(adjacent_frames)

        corrected_prior = transfer_voxel_visibility(adjacent_frames, pts1, cell_size=cfg['cell_size'])

        # Correct priors and store them separately
        ego_prior1[corrected_prior == 1] = -1  # nbr of points from adjacent frame is present in same space
        sequence.store_feature(ego_prior1, idx=frame, name='corrected_ego_prior_label')
        sequence.store_feature(corrected_prior, idx=frame, name='correction_prior')

        visibility_prior = sequence.get_feature(idx=frame, name='visibility_prior')
        visibility_prior[corrected_prior == 1] = -1
        sequence.store_feature(visibility_prior, idx=frame, name='corrected_visibility_prior')

def project_dynamic_label_to_cell(sequence, cfg):

    Bev = BEV(cell_size=(cfg['cell_size'][0], cfg['cell_size'][1]))

    for frame in tqdm(range(len(sequence))):

        # if frame != 4481: continue

        prior_label = - np.ones(sequence.get_feature(idx=frame, name='lidar').shape[0], dtype=np.int32)

        # this should be rewritten anyway
        ego_label = sequence.get_feature(idx=frame, name='corrected_ego_prior_label')
        # ego_label = sequence.get_feature(idx=frame, name='ego_prior_label')
        prior_label[ego_label == 1] = 1  # dynamic object




        pts1 = sequence.get_feature(frame, name='lidar')
        Bev.create_bev_template_from_points(pts1)

        prior_static_mask = sequence.get_feature(idx=frame, name='prior_static_mask')
        prior_label[prior_static_mask == cfg['required_static_time']] = 0  # determistically static

        deselect_static_grid = Bev.generate_bev(pts1[prior_static_mask == -1], 1)
        deselect_static_mask = Bev.transfer_features_to_points(pts1, deselect_static_grid)
        prior_label[deselect_static_mask == 1] = -1     # throw away accumulated static points, that are in the freespace (truck, slow, stopping objects)

        # Dynamic
        visibility_label = sequence.get_feature(idx=frame, name='visibility_prior')
        corrected_visibility_label = sequence.get_feature(idx=frame, name='corrected_visibility_prior')
        visibility_mask = (visibility_label == 1) & (corrected_visibility_label == 1)
        prior_label[visibility_mask] = 1

        road_label = sequence.get_feature(idx=frame, name='road_proposal')
        prior_label[road_label == True] = 0

        # Now prior label is preloaded for processing


        # final_prior = sequence.get_feature(frame, name='final_prior_label')

        indices = prior_label.argsort()  # dynamic labels sorted as last, so it will appear in grid with preference

        lowest_indices = pts1[:, 2].argsort()[::-1]

        lowest_height_grid = Bev.generate_bev(pts1[lowest_indices], pts1[lowest_indices, 2])




        dynamic_grid = Bev.generate_bev(pts1[indices], prior_label[indices])

        # add double z_var to get rid of wrong reflections?

        prop_prior = Bev.transfer_features_to_points(pts1, dynamic_grid)
        prop_height = Bev.transfer_features_to_points(pts1, lowest_height_grid)
        # add heightest point to separate trees above
        # prop_height_lower_boundary = Bev


        # deselect the dynamic, when it the cell has z_var < z_var for road
        deselect_dynamic = pts1[prior_label == 1, 2] > prop_height[prior_label == 1] + cfg['z_var_road']
        deselect_dynamic = 2 * deselect_dynamic - 1         # 1 keep dynamic, -1 to unlabelled

        prior_label[prior_label == 1] = deselect_dynamic


        keep_above_dynamic_mask = pts1[:, 2] >= prop_height + cfg['z_var_road'] + 0.1

        # dynamic_indices = np.argwhere(prop_prior == 1)[:, 0]

        final_dynamic = ((prop_prior == 1) & keep_above_dynamic_mask) | (prior_label == 1)
        final_dynamic = final_dynamic.astype('int')
        final_dynamic[final_dynamic != True] = -1
        final_dynamic[prior_label == 0] = 0

        # Store final gen label!
        sequence.store_feature(final_dynamic, frame, name='final_prior_label')


def store_final_priors_in_mos_format(sequence):

    for frame in tqdm(range(len(sequence))):

        prior_label = sequence.get_feature(frame, name='final_prior_label')

        save_path = sequence.sequence_path + f'/train_prior_mos_labels/{frame:06d}.label'
        store_mos_format(prior_label, save_path)

if __name__ == '__main__':
    from motion_supervision import constants as C
    from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence
    from my_datasets.kitti.semantic_kitti import SemanticKitti_Sequence
    import sys

    # loop over used datasets, when ready
    seq = int(sys.argv[1])

    # seq = 0
    # sequence = Argoverse2_Sequence(sequence_nbr=seq)
    sequence = SemanticKitti_Sequence(sequence_nbr=seq)

    # generate_road_segmentation_by_var(sequence, cfg=C.cfg)

    # print("Generation of static points")
    # generate_static_points_from_sequence(sequence, C.cfg)

    # print("Generation of ego priors")
    # run_ego_priors(sequence, C.cfg)

    # print("Generation of freespace")
    # generate_freespace_from_sequence(sequence, C.rays)
