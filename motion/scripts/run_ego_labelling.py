from tqdm import tqdm

from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence
from my_datasets import paths

from motion_supervision.transfer import Sequence_with_Object, Dynamic_Instance

from my_datasets.visualizer import *



def find_objects_by_ego(sequence, pts_source, frame=30):


    Ego = Dynamic_Instance()
    ego_poses = sequence.get_ego_poses()
    Ego.full_update(pose=ego_poses)
    Ego.box = sequence.ego_box
    Ego.compute_trajectory(from_modality='pose')


    Scene = Sequence_with_Object(sequence, Ego, pts_source=pts_source)

    objects_pts = Scene.annotate_movable(frame)

    return objects_pts

def find_points_by_ego(sequence, pts_source, frame):
    Ego = Dynamic_Instance()
    ego_poses = sequence.get_ego_poses()
    Ego.full_update(pose=ego_poses)
    Ego.box = sequence.ego_box
    Ego.compute_trajectory(from_modality='pose')



if __name__ == '__main__':
    Jackal = {'x': 0.,
              'y': 0.,
              'z': 0.,
              'l': 0.5,
              'w': 0.4,
              'h': 0.4,
              'yaw': 0.}

    # match only inside the intersections
    # sequence = Delft_Sequence()
    # torch.cuda.set_device(2)
    DATASET_PATH = machine_paths.argoverse2

    sequence = Argoverse2_Sequence(DATASET_PATH, sequence_nbr=0)
    store_path = sequence.sequence_path + '/objects/'
    os.makedirs(store_path, exist_ok=True)
    object_store_id = 0
    # run everything on RCI, store which points and from which time of ego

    Ego = Dynamic_Instance()
    ego_poses = sequence.get_ego_poses()
    Ego.full_update(pose=ego_poses)
    Ego.box = sequence.ego_box
    Ego.compute_trajectory(from_modality='pose')


    for frame in tqdm(range(33,len(sequence))):

        pts = sequence.get_global_pts(frame, 'lidar')
        dynamic_mask, time_mask = Ego.drive_through_objects(pts)
        sequence.store_feature(dynamic_mask, frame, name='ego_dynamic')
        sequence.store_feature(time_mask, frame, name='ego_box_time')
