import numpy as np
import matplotlib.pyplot as plt
import torch.nn

from my_datasets import visualizer
from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence
from scipy.spatial.transform.rotation import Rotation
from motion_supervision.motionflow import numpy_chamfer_distance
from motion_supervision.motionflow import Hungarian_point_matching
from timespace.geometry import calculate_yaw
from sklearn.cluster import DBSCAN

def get_sample_data(sequence, frame=50, direction=+1):

    # function sample points
    pts1 = sequence.get_global_pts(frame, 'lidar')
    pts2 = sequence.get_global_pts(frame + direction, 'lidar')

    tm1 = sequence.get_feature(frame, 'ego_box_time')
    tm2 = sequence.get_feature(frame + direction, 'ego_box_time')

    # find same box id in both times (points from consecutive times are inside one ego position)
    shared_tm = np.intersect1d(np.unique(tm1[tm1 > -1]), np.unique(tm2[tm2 > -1]))

    # ego motion compensation - working
    pose1 = sequence.get_feature(frame, 'pose')
    pose2 = sequence.get_feature(frame + direction, 'pose')

    tmp_pts1 = pts1.copy()
    tmp_pts1[:,3] = 1

    tmp_pts2 = pts2.copy()
    tmp_pts2[:, 3] = 1

    back_pts1 = (np.linalg.inv(pose1) @ tmp_pts1.T).T
    back_pts2 = (np.linalg.inv(pose1) @ tmp_pts2.T).T

    return back_pts1, back_pts2, tm1, tm2


def motion_from_medians(c_p, n_p, use_chamfer=False, yaw_limits=(-15, 15), plot=False):
    '''
    Assign medians as center of objects and calculate yaw angle between. Might be subject to smoothness
    afterwards. If use_chamfer == True, the function applies rigid fitting by chamfer distanc minimization
    :param c_p:
    :param n_p:
    :param use_chamfer:
    :param yaw_limits: in degrees
    :param plot:
    :return:
    '''
    med1 = np.median(c_p, axis=0)
    med2 = np.median(n_p, axis=0)

    diff = med2 - med1

    tr_p = c_p + diff  # to second center
    tr_p_origin = tr_p - med2  # to origin for rotation

    diff = med2 - med1
    yaw_from_meds = np.arctan2(diff[1], diff[0])
    yaw_degree = 180 * yaw_from_meds / np.pi

    best_yaw = yaw_from_meds

    if use_chamfer:
        min_chamf_dist = 10000
        best_yaw = yaw_from_meds

        for tmp_yaw in range(int(yaw_degree) + yaw_limits[0], int(yaw_degree) + yaw_limits[1]):
            tmp_yaw = tmp_yaw / 180 * np.pi  # to rads
            R_mat = Rotation.from_rotvec(np.array((0, 0, tmp_yaw))).as_matrix()
            rotated_coors = tr_p_origin[:, :3] @ R_mat.T
            rotated_coors += med2[:3]  # back to second objects center

            chamf_dist = numpy_chamfer_distance(n_p[:, :3], rotated_coors)

            if min_chamf_dist > chamf_dist:
                min_chamf_dist = chamf_dist
                best_yaw = tmp_yaw

                if plot:
                    plt.plot(c_p[:, 0], c_p[:, 1], '.b')
                    plt.plot(n_p[:, 0], n_p[:, 1], '.r')
                    plt.plot(rotated_coors[:, 0], rotated_coors[:, 1], '.g')

                    plt.plot(med2[0], med2[1], '*r')
                    plt.plot(med1[0], med1[1], '*b')
                    plt.plot([med1[0], med2[0]], [med1[1], med2[1]], '-->')
                    plt.annotate(f"{yaw_degree:.2f} deg", med1[:2] + (0.01, 0))
                    plt.axis('equal')
                    plt.title(f"Motion from Medians: Yaw {tmp_yaw * 180 / np.pi:.3f} Deg t Chamf: {chamf_dist:.3f}")
                    plt.show()
                    plt.close()

    return best_yaw, med1, med2


def plot_rigid_flow(c_p, rigid_flow):
    fig, ax = plt.subplots()
    ax.plot(c_p[:, 0], c_p[:, 1], 'bo')
    ax.plot(n_p[:, 0], n_p[:, 1], 'ro')

    for i in range(len(c_p)):
        ax.annotate("", xy=(c_p[i, 0] + rigid_flow[0], c_p[i, 1] + + rigid_flow[1]), xytext=(c_p[i, 0], c_p[i, 1]),
                    arrowprops=dict(facecolor='green'))
    plt.axis('equal')
    plt.title('Rigid flow from mean flow vector')
    plt.show()


def plot_two_point_sets(c_p, n_p):
    '''
    Plots two objects, their medians and connected yaw angle with respect to the origin of xyz
    :param c_p: first point cloud
    :param n_p: second point cloud
    :return:
    '''
    med1 = np.median(c_p, axis=0)
    med2 = np.median(n_p, axis=0)

    diff = med2 - med1
    yaw_from_meds = np.arctan2(diff[1], diff[0])
    yaw_degree = 180 * yaw_from_meds / np.pi

    plt.plot(c_p[:, 0], c_p[:, 1], '.b')
    plt.plot(n_p[:, 0], n_p[:, 1], '.r')
    plt.plot(c_p[:, 0] + diff[0], c_p[:, 1] + diff[1], '.g')

    plt.plot(med2[0], med2[1], '*r')
    plt.plot(med1[0], med1[1], '*b')
    plt.plot([med1[0], med2[0]], [med1[1], med2[1]], '-->')
    plt.annotate(f"{yaw_degree:.2f} deg", med1[:2] + (0.01, 0))
    plt.axis('equal')
    plt.show()
    plt.close()

class ToyMotionFlowNet(torch.nn.Module):
    def __init__(self, nbr_pts):
        super().__init__()
        self.weights = torch.rand(nbr_pts, 3, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        ''' Do not take batch size. Just to try all losses etc. Is fully expresive on one flow example'''
        out = x  * self.weights
        return out

def match_points_unique_cluster(c_p, n_p, yaw_tolerance=10, min_pts=2, plot=False):
    '''
    :param c_p: first pts
    :param n_p: next pts
    :param yaw_tolerance: Allowed variance of yaw angles that one cluster can have
    :param min_pts: minimum point of cluster to be considered
    :param plot:
    :return: Flow vector N x [x,y,z,mask]
    '''
    flow_vector = np.zeros((len(c_p), 4), dtype=float)
    c_p[:, 2] *= 3  # this is to enforce similar altitude and scale z - rigidity :D
    match_ids = Hungarian_point_matching(c_p, n_p, plot=False)
    c_p[:, 2] /= 3
    yaws = calculate_yaw(c_p, n_p[match_ids])  # cluster
    dist = np.abs(n_p[match_ids, :3] - c_p[:, :3]).mean(1)  # cluster as well?
    # todo configurable with other distance clustering, etc. Or maybe calculate final loss
    # todo apply rigidity as well - one flow?
    # todo eliminate different z - do it inside the hungarian algorithm
    model = DBSCAN(eps=yaw_tolerance,
                   min_samples=min_pts)  # 10degrees for yaws, 1 for min points considered a object
    same_direction = model.fit(yaws[:, None] + 360).labels_ # 360 will solve border angles? Seems like it

    if np.all(same_direction == -1):
        return flow_vector

    u, c = np.unique(same_direction[same_direction!= -1], return_counts=True) # eliminate noise cluster
    biggest_cluster = u[np.argmax(c)]

    # get only selected points
    new_pts = c_p[same_direction == biggest_cluster]
    new_pts2 = n_p[match_ids]
    new_pts2 = new_pts2[same_direction == biggest_cluster]

    velocity = new_pts2[:,:3] - new_pts[:,:3]

    flow_vector[same_direction == biggest_cluster, :3] = velocity
    flow_vector[same_direction == biggest_cluster, 3] = 1

    valid_flow = flow_vector[:, 3] == 1
    rigid_flow = flow_vector[valid_flow].mean(0)  # can be done more inteligently
    if plot:
        print(yaws[same_direction == biggest_cluster])
        visualizer.visualize_connected_points(new_pts, new_pts2, title=f'Hungarian with unique clusters, yaw tolerance - {yaw_tolerance}')


    return flow_vector

if __name__ == '__main__':
    sequence = Argoverse2_Sequence(sequence_nbr=0)

    # get pcls ego-motion compensated todo apply ego-compensation to basic processing
    direction = +1
    frame = 50

    back_pts1, back_pts2, tm1, tm2 = get_sample_data(sequence, frame, direction)

    # get overlap of two times inside one box
    # todo solve overlapping points from multiple bounding boxes
    shared_tm = np.intersect1d(np.unique(tm1[tm1 > -1]), np.unique(tm2[tm2 > -1]))


    pts1 = back_pts1
    pts2 = back_pts2

    # check if multiple instances inside one ego box
    # two people inside one box in that sequence
    gt_id_mask1 = sequence.get_feature(frame, 'id_mask')

    # clustering uvnitr boxu i na motion featurech z Hungarianu. Ty body s jinym flow pak asi vypadnou
    # + do toho hungarianu zaradit i rigiditu? To chce nejak vymyslet, asi cycle loss?


    frame_flow = np.zeros((pts1.shape[0], 4), dtype=float)

    for box_position in shared_tm:

        current_indices = np.argwhere(tm1 == box_position)[:, 0]

        c_p = pts1[current_indices].copy()
        n_p = pts2[tm2==box_position].copy()


        if len(c_p) < 2: continue
        print(c_p.shape, n_p.shape)

        # hack to develop, should be treated better when production
        # such as swap both but reverse flow
        #todo forth and back? and less points in next frame. Subsample the first one :D
        if len(c_p) > len(n_p):
            subsample_ids = np.random.choice(len(c_p), len(n_p))
            c_p = c_p.copy()[subsample_ids]
        else:
            subsample_ids = np.arange(len(c_p))

        flow_vector = match_points_unique_cluster(c_p, n_p, yaw_tolerance=10, min_pts=2, plot=True)

        frame_flow[current_indices[subsample_ids]] = flow_vector

        # calculate rigid flow/ unified flow
        valid_flow = flow_vector[:,3] == 1
        rigid_flow = flow_vector[valid_flow].mean(0)    # can be done more inteligently

        # center = c_p.mean(0)

        motion_from_medians(c_p, n_p, use_chamfer=True, yaw_limits=(-20,20), plot=False)

