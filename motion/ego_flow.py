from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence
from motion_supervision.connect_points import match_points_unique_cluster
from motion_supervision.transfer import Dynamic_Instance

from my_datasets.visualizer import *

config = {'exp_name' : "Toy",
          'matching_algo' : 'match_points_unique_cluster',
          'pts_source' : 'lidar',
          }


def drive_through_objects(pts, ego_box, secure_margin=0.25, add_z=0):

    object_mask = np.zeros(pts.shape[0], dtype=bool)

    upper_half_points = box_utils.get_point_mask(pts, ego_box, z_add=(0, add_z)) & (
            pts[:, 2] > ego_box[2] - ego_box[5] * secure_margin)

    object_mask[upper_half_points] = True

    return object_mask

class Ego_Flow():

    def __init__(self, sequence, cfg):
        self.sequence = sequence
        self.cfg = cfg

        self.Ego = Dynamic_Instance()
        ego_poses_dict = sequence.get_ego_poses()
        self.Ego.full_update(pose=ego_poses_dict)
        self.Ego.box = sequence.ego_box
        self.Ego.compute_trajectory(from_modality='pose')
        # self.Ego.Trajectory.plot()


    def one_frame(self, frame1, frame2):

        pts1 = self.sequence.get_global_pts(frame1, name=self.cfg['pts_source'])
        pts2 = self.sequence.get_global_pts(frame2, name=self.cfg['pts_source'])

        frame_flow = np.zeros((pts1.shape[0], 4), dtype=float)
        frame_rigid_flow = np.zeros((pts1.shape[0], 4), dtype=float)

        print('here')
        # as you are used to
        dyn1, road1, tm1 = self.Ego.drive_through_objects(pts1, min_time_box=frame1-500, max_time_box=frame1+500)
        dyn2, road2, tm2 = self.Ego.drive_through_objects(pts2, min_time_box=frame1-500, max_time_box=frame1+500)
        print('after here')

        # todo tmp
        self.sequence.store_feature(tm1, idx=frame1, name='ego_box_time')
        self.sequence.store_feature(tm2, idx=frame2, name='ego_box_time')

        self.sequence.store_feature(dyn1, idx=frame1, name='ego_dyn')
        self.sequence.store_feature(dyn2, idx=frame2, name='ego_dyn')

        self.sequence.store_feature(road1, idx=frame1, name='ego_road')
        self.sequence.store_feature(road2, idx=frame2, name='ego_road')
        # get synchronized point clouds next to each other - to have same metric as well
        syn_pts1, syn_pts2 = self.sequence.get_two_synchronized_frames(frame1, frame2, pts_source=self.cfg['pts_source']) #

        # Find points inside the same ego box from both point clouds
        shared_tm = np.intersect1d(np.unique(tm1[tm1 > -1]), np.unique(tm2[tm2 > -1]))

        for box_position in shared_tm:
            # Store indices for backprop
            current_indices = np.argwhere(tm1 == box_position)[:, 0]

            # sample the points for current bounding box
            c_p = syn_pts1[current_indices].copy()
            n_p = syn_pts2[tm2 == box_position].copy()

            # remove low points
            if len(c_p) < 2 or len(c_p) + len(n_p) > 200: continue # hungarian is unfeasable, kernel would work
            # print(c_p.shape, n_p.shape)

            # such as swap both but reverse flow
            # todo forth and back? and less points in next frame. Subsample the first one :D
            # todo adjust z in matching
            if len(c_p) > len(n_p):
                subsample_ids = np.random.choice(len(c_p), len(n_p))
                c_p = c_p.copy()[subsample_ids]
            else:
                subsample_ids = np.arange(len(c_p))

            flow_vector = match_points_unique_cluster(c_p, n_p, yaw_tolerance=10, min_pts=2, plot=False)

            frame_flow[current_indices[subsample_ids]] = flow_vector

            # calculate rigid flow/ unified flow
            # todo
            # valid_flow = flow_vector[:, 3] == 1
            # rigid_flow = flow_vector[valid_flow].mean(0)  # can be done more inteligently
            # frame_rigid_flow[current_indices[subsample_ids]] = rigid_flow


        return frame_flow, frame_rigid_flow
        # separate latter
        # dynamic1 = drive_through_objects(pts1, ego_box=self.Ego.Trajectory.boxes[frame1])
        # dynamic2 = drive_through_objects(pts2, ego_box=self.Ego.Trajectory.boxes[frame2])

    def run_ego_flow(self):
        for frame in range(len(self.sequence) - 1):
            frame_flow, rigid_flow = self.one_frame(frame, frame+1)

            self.sequence.store_feature(frame_flow, frame, name='ego_flow')
            self.sequence.store_feature(rigid_flow, frame, name='ego_rigid_flow')

        # last is not evaluated
        last_pts = self.sequence.get_global_pts(len(self.sequence) - 1, name=self.cfg['pts_source'])
        self.sequence.store_feature(np.zeros((last_pts.shape[0], 4)), len(self.sequence) - 1, name='ego_flow')
        self.sequence.store_feature(np.zeros((last_pts.shape[0], 4)), len(self.sequence) - 1, name='ego_rigid_flow')

def main(sequence): # to dataset

    Ego = Dynamic_Instance()
    ego_poses_dict = sequence.get_ego_poses()
    Ego.full_update(pose=ego_poses_dict)
    Ego.box = sequence.ego_box
    Ego.compute_trajectory(from_modality='pose')

    Flow_Runner = Ego_Flow(sequence=sequence, cfg=config)
    # Flow_Runner.one_frame(frame1=33, frame2=34)
    Flow_Runner.run_ego_flow()

if __name__ == "__main__":
    # sys arg will be nbr of processes. Here it would calculate "Efficient Distribution"
    if len(sys.argv) > 1:
        nbr_of_processes = int(sys.argv[1])
    else:
        nbr_of_processes = 1



    sequence = Argoverse2_Sequence(sequence_nbr=1)
    fr = 33
    pts1 = sequence.get_feature(fr, 'lidar')
    pts2 = sequence.get_feature(fr+1, 'lidar')
    box1 = sequence.get_feature(fr, 'boxes')
    box2 = sequence.get_feature(fr+1, 'boxes')
    ego1 = sequence.get_feature(fr, 'pose')
    ego2 = sequence.get_feature(fr+1, 'pose')

    from timespace.trajectory import construct_transform_matrix
    from timespace import box_utils

    def calculate_flow(pts1, box1, box2, ego_pose1, ego_pose2, move_threshold=0.05):   # pts and boxes in local frame
        flow = np.zeros((pts1.shape[0], 4), dtype=float)
        dynamic = np.zeros(pts1.shape[0])

        tmp_pts1 = pts1.copy()
        tmp_pts1[:,3] = 1
        T2_to_T1 = np.linalg.inv(ego_pose1) @ ego_pose2

        pts1_in_pts2 =  tmp_pts1 @ T2_to_T1.T

        flow[:, :3] = - pts1_in_pts2[:,:3] + pts1[:,:3]   # rigid flow from ego-motion


        # per object flow
        id_box_dict = {box['uuid'] : box for box in box2}
        for one_box in box1:

            box1_uuid = one_box['uuid']

            if box1_uuid not in id_box_dict.keys():
                pts_in_box = box_utils.get_point_mask(pts1, one_box)
                flow[pts_in_box, 3] = -1    # not available
                continue    # ended here

            second_box = id_box_dict[box1_uuid] # solve

            # find the same
            pts_in_box = box_utils.get_point_mask(pts1, one_box)
            box1_T_mat = construct_transform_matrix(one_box['rotation'], one_box['translation'])
            box2_T_mat = construct_transform_matrix(second_box['rotation'], second_box['translation'])

            obj_shift = box2_T_mat @ np.linalg.inv(box1_T_mat)
            # separate points of object and background (rigid_flow is already included in annotation)
            tmp_obj_pts = pts1[pts_in_box].copy()
            tmp_obj_pts[:,3] = 1

            transformed_obj_pts = tmp_obj_pts[:,:4] @ obj_shift.T
            shift_flow = transformed_obj_pts[:,:3] - pts1[pts_in_box,:3]


            # Dynamic
            box1_dyn = box1_T_mat[:,-1] @ ego_pose1.T
            box2_dyn = box2_T_mat[:,-1] @ ego_pose2.T

            velocity = box1_dyn - box2_dyn
            vector_velocity = np.sqrt((velocity ** 2)).sum(0)

            if vector_velocity > move_threshold and one_box['potentially_dynamic']:
                # todo continue this ^
                dynamic[pts_in_box] = 1

            flow[pts_in_box, :3] = shift_flow
            flow[pts_in_box, 3] = 1

        # produce the dynamic annotation as well here
        return pts1_in_pts2, flow, dynamic

    reverse_pts1, flow, dynamic = calculate_flow(pts1, box1, box2, ego_pose1=ego1, ego_pose2=ego2)
    # for vis
    flow[:, 3] = 1
    glob_pts1 = sequence.get_global_pts(fr, 'lidar')
    glob_pts2 = sequence.get_global_pts(fr+1, 'lidar')
    # visualizer_flow3d(pts1, pts2, flow)
    # visualizer_flow3d(glob_pts1, glob_pts2, flow)
    visualize_points3D(pts1, dynamic)



    # for seq_nbr in tqdm(range(0,700)):
    #     sequence = Argoverse2_Sequence(sequence_nbr=seq_nbr)
    #
    #     ego_flow_path = sequence.sequence_path + '/ego_flow'
    #     print(seq_nbr, ego_flow_path)
    #
    #     if os.path.exists(ego_flow_path):
    #         continue
    #     else:
    #         os.makedirs(ego_flow_path, exist_ok=True)
    #         main(sequence)
