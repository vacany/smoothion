import torch

from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence
from motion_supervision.generate_priors import run_instance_generation



class Ego_Loss(torch.nn.Module):    # working name

    def __init__(self, reduction='mean'):
        super().__init__()
        self.CrossEntropy = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)

    def forward(self, prediction, marked_by_ego):   # marked by ego: -1 - dont know, 0 - static, 1 - dynamic
        if len(prediction.shape) == 2:
            prediction = prediction.permute(1,0)
            prediction = prediction.unsqueeze(0)    # adding BS dim


        if len(marked_by_ego.shape) == 1:
            marked_by_ego = marked_by_ego.unsqueeze(0) # adding BS dim

        loss = self.CrossEntropy(prediction, marked_by_ego)
        return loss


if __name__ == "__main__":
    sequence = Argoverse2_Sequence(sequence_nbr=1)

    accum_pts = run_instance_generation(sequence)




    # visualize accum_pts by unique value in clus in for loop
    # for i in np.unique(clus):
    #     if i == -1: continue
    #     visualize_points3D(np.array(accum_pts)[clus==i])

    # for seq_nbr in tqdm(range(0,700)):
    #     break
    #     sequence = Argoverse2_Sequence(sequence_nbr=seq_nbr)
    #
    #     ego_flow_path = sequence.sequence_path + '/ego_cluster'
    #     print(seq_nbr, ego_flow_path)
    #
    #     if os.path.exists(ego_flow_path):
    #         continue
    #     else:
    #         os.makedirs(ego_flow_path, exist_ok=True)
    #         run_instance_generation(sequence)



    #     # loss_function = Ego_Loss(reduction='sum')
    #     # CE_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    #
    #     pts = sequence.get_global_pts(idx=frame, name='lidar')
    #
    #     # eliminate points in point cloud by radius small
    #     # transform to ego frame and eliminate points in ego frame by radius small
    #     pose = sequence.get_feature(idx=frame, name='pose')
    #     sensor_valid_mask = np.linalg.norm(pts[:, :3] - pose[:3,-1], axis=1) > 2
    #
    #
    #     ego_boxes = sequence.get_ego_boxes()
    #
    #     ego_label, tm = mark_by_driving(pts, ego_boxes, return_time_mask=True)
    #
    #     # create Data for instance segmentation
    #     instance_pred, valid_mask, prediction = cluster_marked_pts(pts, tm, min_samples=1)
    #
    #     if valid_mask is None:
    #         continue
    #     # set weights as prediction
    #     # weights = torch.rand([instance_pred.shape[0], 10], dtype=torch.float) * 0.2
    #     # weights.requires_grad_(True)
    #     # weights = torch.nn.Parameter(weights)
    #     # optimizer = torch.optim.Adam([weights], lr=0.01)
    #     # # transform prediction to label format
    #     # for e in range(20):
    #     #     instance_ego_label = torch.tensor(prediction.copy(), dtype=torch.long)
    #     #     loss_inst = CE_loss(weights, instance_ego_label)
    #     #     loss_inst.backward()
    #     #
    #     #     instance_label = torch.argmax(weights, dim=1)
    #     #     lookat = np.array([1608, 260, 15])
    #     #     plot_points3d(pts, instance_label.numpy() / 11, lookat=lookat, title='instance label',
    #     #                   save=f'{os.path.expanduser("~")}/projects/delft/overleaf/instance_examples/{e:03d}.png')
    #     #
    #     #     optimizer.step()
    #     #     optimizer.zero_grad()
    #     #     print(loss_inst.item())
    #
    #
    #
    #     valid_mask[sensor_valid_mask==False] = False
    #
    #     id_mask = sequence.get_feature(frame, 'id_mask')
    #
    #     pred_iou, pred_matched, confidence, n_gt_inst = eval_segm(id_mask[valid_mask], instance_pred[valid_mask, :])
    #
    #     _ = [iou_list.append(iou) for iou in pred_iou]
    #
    #     print(f"Frame: {frame} Final IOU: {np.mean(iou_list)}")
