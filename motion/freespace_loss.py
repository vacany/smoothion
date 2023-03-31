import socket
import sys

from datasets.waymo.waymo import Waymo_Sequence
from datasets.visualizer import *

from motion_supervision.visibility import transfer_voxel_visibility
import torch
from pytorch3d.ops.knn import knn_points

from datasets.paths import TMP_VIS

if socket.gethostname().startswith('Patrik'):
    import sys

    e = int(sys.argv[1])

    data_dict = np.load(TMP_VIS + f'/{e}_flow.npz', allow_pickle=True)

    freespace = data_dict['freespace']
    # freespace[:, 2] = freespace[:, 3]
    # visualize_multiple_pcls(*[data_dict['p_i'],  data_dict['p_i'][:,:3] + data_dict['flow'], data_dict['p_j']])

    visualize_flow3d(data_dict['p_i'], data_dict['p_j'], data_dict['flow'])

    # visualize_points3D(data_dict['p_i'], data_dict['p_i'][:, 2] > 0)
    # visualize_points3D(data_dict['p_i'], data_dict['loss'])
    sys.exit('Done')

def torch_voxelized_space(p_i, flow, freespace, cell_size):
    # should be also written for gpu
    cost_values = []

    p_i_flow = p_i + flow

    for bs in range(len(p_i_flow)):

        # rays - to function
        Nbr_pts = len(p_i[bs])
        nbr_inter = 10
        ray_pts = torch.stack([p_i[bs] + (i + 1) / nbr_inter * flow[bs] for i in range(nbr_inter)], dim=0)


        full_pts = torch.cat([p_i_flow[bs, :, :3], freespace[bs, :, :3], ray_pts.view(-1, 3)], dim=0)

        x_max, y_max, z_max = torch.max(full_pts, dim=0)[0]
        x_min, y_min, z_min = torch.min(full_pts, dim=0)[0]


        voxel_map = torch.zeros((( (x_max - x_min) / cell_size[0] + 2).long(),
                                 ( (y_max - y_min) / cell_size[1] + 2).long(),
                                 ( (z_max - z_min) / cell_size[2] + 2).long()), dtype=torch.float, device=p_i_flow.device)


        p_i_flow_idx = torch.round( (p_i_flow[bs, :, :3] - torch.stack((x_min, y_min, z_min))) / cell_size).long()
        inter_flow_idx = torch.round( (ray_pts.view(-1, 3) - torch.stack((x_min, y_min, z_min))) / cell_size).long()
        freespace_idx = torch.round( (freespace[bs,:, :3] - torch.stack((x_min, y_min, z_min))) / cell_size).long()


        voxel_map[freespace_idx[:, 0], freespace_idx[:, 1], freespace_idx[:, 2]] = freespace[bs, :, 3]

        cost_value = voxel_map[p_i_flow_idx[:, 0], p_i_flow_idx[:, 1], p_i_flow_idx[:, 2]]
        ray_cost_value = voxel_map[inter_flow_idx[:, 0], inter_flow_idx[:, 1], inter_flow_idx[:, 2]]

        # take maximum cost from all rays
        inter_cost_value = torch.max(ray_cost_value.view(nbr_inter, Nbr_pts), dim=0)[0]
        # breakpoint()

        # cost_values.append(cost_value)
        cost_values.append(inter_cost_value)

    return torch.stack(cost_values)



sequence = Waymo_Sequence(1)
# load data
frame = 0
p_i = torch.tensor(sequence.get_feature(idx=frame, name='lidar')[None, :,:3], dtype=torch.float).cuda()
p_j = torch.tensor(sequence.get_feature(idx=frame+1, name='lidar')[None, :,:3], dtype=torch.float).cuda()

p_i = p_i[p_i[..., 2] > 0.3][None, ...]
p_j = p_j[p_j[..., 2] > 0.3][None, ...]

freespace = sequence.get_feature(idx=frame+1, name='accum_freespace')[None, ...] # shorter freespace so far
freespace = torch.tensor(freespace, dtype=torch.float).cuda()

distance_from_sensor = torch.linalg.norm(freespace[..., :3], dim=2)
freespace_cost = 1 / torch.sqrt(distance_from_sensor)
freespace[..., 3] = freespace_cost



flow = torch.randn((1, p_i.shape[1], 3)).cuda().requires_grad_()
optimizer = torch.optim.Adam([flow], lr=0.1)

# visualize_multiple_pcls(*[p_i, p_i[:,:3] + flow, freespace, p_j])
# with torch.no_grad():
#     cost_value = torch_voxelized_space((p_i.detach() + flow.detach())[0], freespace)
# visualize_points3D(p_i + flow, cost_value)

cell_size = torch.tensor((0.2,0.2,0.2)).cuda()

for e in range(1000):

    with torch.no_grad():
        cost_value = torch_voxelized_space(p_i.detach(), flow.detach(), freespace, cell_size)

                           # flow magnitude        # euclidian distance from position
    loss_freespace = - (p_i[..., :2] + flow[..., :2]).norm(dim=2) * cost_value #+ flow.norm(dim=2) * cost_value  # this will be changed in BS


    subsample_idx = torch.randint(0, p_j.shape[1], (p_i.shape[1], 1))[:, 0]
    # Nn loss
    x_nn = knn_points(p_i + flow, p_j[:, subsample_idx, :], K=1, norm=1)
    y_nn = knn_points(p_j[:, subsample_idx, :], p_i + flow, K=1, norm=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    nn_loss = (cham_x + cham_y) / 2
    nn_loss = nn_loss.mean()

    # keep mask? beware of mean value at the end - try both versions
    loss = 2 * loss_freespace.mean() + nn_loss
    # loss = nn_loss
    # decreasing freespace wont work on points inside freespace, as the building for example
    # you need to shift the flow outside completely. Maybe add randomly or NN artificial flow to get outside and then match


    loss.backward()

    print(e, loss, torch.sum(cost_value > 0), flow.max(), flow.min())

    vis_dict = {'p_i' : p_i[0].detach().cpu().numpy(),
                'p_j' : p_j[0].detach().cpu().numpy(),
                'flow' : flow[0].detach().cpu().numpy(),
                'freespace' : freespace[0].detach().cpu().numpy(),
                'loss' : loss_freespace[0].detach().cpu().numpy()
                }

    np.savez(TMP_VIS + f'{e}_flow.npz', **vis_dict)

    optimizer.step()
    optimizer.zero_grad()




