import numpy as np
from scipy import sparse
import torch
import pandas as pd

# mostly taken from OGC

def eval_segm(segm, mask, ignore_npoint_thresh=0):
    """
    :param segm: (N,).
    :param segm_pred: (N, K).
    :return:
        pred_iou: (N,).
        pred_matched: (N,).
        confidence: (N,).
        n_gt_inst: An integer.
    """
    segm_pred = np.argmax(mask, axis=1)
    _, segm, gt_sizes = np.unique(segm, return_inverse=True, return_counts=True)
    pred_ids, segm_pred, pred_sizes = np.unique(segm_pred, return_inverse=True, return_counts=True)
    n_gt_inst = gt_sizes.shape[0]
    n_pred_inst = pred_sizes.shape[0]
    mask = mask[:, pred_ids]

    # Compute Intersection
    intersection = np.zeros((n_gt_inst, n_pred_inst))
    for i in range(n_gt_inst):
        for j in range(n_pred_inst):
            intersection[i, j] = np.sum(np.logical_and(segm == i, segm_pred == j))

    # Ignore too small GT objects
    ignore_gt_ids = np.where(gt_sizes < ignore_npoint_thresh)[0]

    # An FP is not penalized, if mostly overlapped with ignored GT
    pred_ignore_ratio = np.sum(intersection[ignore_gt_ids], axis=0) / pred_sizes
    invalid_pred = (pred_ignore_ratio > 0.5)

    # Kick out predictions' area intersectioned with ignored GT
    pred_sizes = pred_sizes - np.sum(intersection[ignore_gt_ids], axis=0)
    valid_pred = np.logical_and(pred_sizes > 0, np.logical_not(invalid_pred))

    intersection = np.delete(intersection, ignore_gt_ids, axis=0)
    gt_sizes = np.delete(gt_sizes, ignore_gt_ids, axis=0)
    n_gt_inst = gt_sizes.shape[0]

    intersection = intersection[:, valid_pred]
    pred_sizes = pred_sizes[valid_pred]
    mask = mask[:, valid_pred]
    n_pred_inst = pred_sizes.shape[0]

    # Compute confidence scores for predictions
    confidence = np.zeros((n_pred_inst))
    for j in range(n_pred_inst):
        inst_mask = mask[segm_pred == j, j]
        confidence[j] = np.mean(inst_mask)

    # Find matched predictions
    union = np.expand_dims(gt_sizes, 1) + np.expand_dims(pred_sizes, 0) - intersection
    iou = intersection / union
    pred_iou = iou.max(axis=0)
    # In panoptic segmentation, Greedy gives the same result as Hungarian
    pred_matched = (pred_iou >= 0.5).astype(float)
    return pred_iou, pred_matched, confidence, n_gt_inst


def eval_flow(gt_flow, flow_pred, epe_norm_thresh=0.05, eps=1e-10):
    """
    Compute scene flow estimation metrics: EPE3D, Acc3DS, Acc3DR, Outliers3D.
    :param gt_flow: (B, N, 3) torch.Tensor.
    :param flow_pred: (B, N, 3) torch.Tensor.
    :param epe_norm_thresh: Threshold for abstract EPE3D values, used in computing Acc3DS / Acc3DR / Outliers3D and adapted to sizes of different datasets.
    :return:
        epe & acc_strict & acc_relax & outlier: Floats.
    """
    #todo check the shape!
    if type(gt_flow) is np.ndarray and type(flow_pred) is np.ndarray:
        gt_flow = torch.tensor(gt_flow, dtype=torch.float).unsqueeze(0)
        flow_pred = torch.tensor(flow_pred, dtype=torch.float).unsqueeze(0)

    gt_flow = gt_flow.detach().cpu()
    flow_pred = flow_pred.detach().cpu()

    epe_norm = torch.norm(flow_pred - gt_flow, dim=2)
    sf_norm = torch.norm(gt_flow, dim=2)
    relative_err = epe_norm / (sf_norm + eps)
    epe = epe_norm.mean().item()

    # Adjust the threshold to the scale of dataset
    acc_strict = (torch.logical_or(epe_norm < epe_norm_thresh, relative_err < 0.05)).float().mean().item()
    acc_relax = (torch.logical_or(epe_norm < (2 * epe_norm_thresh), relative_err < 0.1)).float().mean().item()
    outlier = (torch.logical_or(epe_norm > (6 * epe_norm_thresh), relative_err > 0.1)).float().mean().item()
    # todo SLIM recall insted of outlier
    return epe, acc_strict, acc_relax,



class IOU(torch.nn.Module):
    def __init__(self, num_classes, exp_name=None, ignore_cls=None, weights=None, clazz_names=None, verbose=False):
        super().__init__()

        self.num_classes = num_classes
        self.exp_name = exp_name

        self.ignore_cls = ignore_cls
        self.verbose = verbose

        self.cm = np.zeros((num_classes, num_classes), 'u8')  # confusion matrix
        self.tps = np.zeros(num_classes, dtype='u8')  # true positives
        self.fps = np.zeros(num_classes, dtype='u8')  # false positives
        self.fns = np.zeros(num_classes, dtype='u8')  # false negatives
        self.weights = weights if weights is not None else np.ones(num_classes)  # Weights of each class for mean IOU
        self.clazz_names = clazz_names if clazz_names is not None else np.arange(num_classes)  # for nicer printing

    def forward(self, batch):
        infer = batch[self.infer_key]
        gt = batch[self.gt_key]

        self.update(labels=gt, predictions=infer)

    def update(self, labels, predictions):
        if type(labels) == torch.Tensor:
            labels = labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()

        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, 1)  # first dimension are probabilities/scores

        if self.ignore_cls is not None:
            mask = labels != self.ignore_cls
            labels = labels[mask]
            predictions = predictions[mask]

        tmp_cm = sparse.coo_matrix(
                (np.ones(np.prod(labels.shape), 'u8'), (labels.flatten(), predictions.flatten())),
                shape=(self.num_classes, self.num_classes)
        ).toarray()

        tps = np.diag(tmp_cm)
        fps = tmp_cm.sum(0) - tps
        fns = tmp_cm.sum(1) - tps
        self.cm += tmp_cm
        self.tps += tps
        self.fps += fps
        self.fns += fns

        if self.verbose:
            self.print_stats()

    def _compute_stats(self, tps, fps, fns):
        with np.errstate(
                all='ignore'):  # any division could be by zero, we don't really care about these errors, we know about these
            precisions = tps / (tps + fps)
            recalls = tps / (tps + fns)
            ious = tps / (tps + fps + fns)

        return precisions, recalls, ious

    def return_stats(self):
        with np.errstate(
                all='ignore'):  # any division could be by zero, we don't really care about these errors, we know about these
            precisions = self.tps / (self.tps + self.fps)
            recalls = self.tps / (self.tps + self.fns)
            ious = self.tps / (self.tps + self.fps + self.fns)

        return precisions, recalls, ious

    def get_precisions(self, interest_clz=[0]):
        precisions, recalls, ious = self._compute_stats(self.tps, self.fps, self.fns)
        res_li = []
        for c in interest_clz:
            res_li.append(np.array((precisions[c], recalls[c], ious[c], self.tps[c], self.fps[c], self.fns[c])))

        res_table = np.stack(res_li)
        return precisions

    def results_to_file(self, file=None):
        precisions, recalls, ious = self._compute_stats(self.tps, self.fps, self.fns)

        ious = np.insert(ious, len(ious), ious.mean())  # mean iou

        # if self.clazz_names[-1] != 'mIou':
        #     self.clazz_names.append('mIou')

        df = pd.DataFrame([precisions[1], recalls[1], ious[1]], index=["Precision", "Recall", "IOU"], columns=[0])
        print(df.to_latex())

        data = {'latex': df.to_latex(),
                'mIou': ious[-1],
                'clazz_names': self.clazz_names[:-1],
                'ious': ious[:-1],
                'precisions': precisions,
                'recalls': recalls,
                }

        if file is not None:
            np.savez(file, **data)

    def print_stats(self, classes=None):
        if classes is None:
            classes = range(self.num_classes)
        precisions, recalls, ious = self._compute_stats(self.tps, self.fps, self.fns)
        # print('\n---\n')
        for c in classes:
            print(
                    f'Class: {str(self.clazz_names[c]):20s}\t'
                    f'Precision: {precisions[c]:.3f}\t'
                    f'Recall {recalls[c]:.3f}\t'
                    f'IOU: {ious[c]:.3f}\t'

                    f'TP: {self.tps[c]:.3f}\t'
                    f'FP {self.fps[c]:.3f}\t'
                    f'FN: {self.fns[c]:.3f}\t'
            )
        # print(f"Mean IoU {ious.mean()}")
        # print('\n---\n')

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), 'u8')
        self.tps = np.zeros(self.num_classes, dtype='u8')
        self.fps = np.zeros(self.num_classes, dtype='u8')
        self.fns = np.zeros(self.num_classes, dtype='u8')

