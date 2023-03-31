import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence
from motion_supervision.metric import IOU

import motion_supervision.constants as C
from my_datasets.visualizer import *

PRIOR_METHODS = ['ego_prior', 'visibility_prior', 'static_prior', 'road_prior']
value = [True, False]

def get_config_frames(options, values):

    combinations = list(itertools.product(values, repeat=len(options)))
    comb_array = np.array(combinations)
    comb_array = comb_array[np.argsort(np.sum(comb_array, axis=1))]

    data_frame = pd.DataFrame(columns=PRIOR_METHODS, data=comb_array)

    return data_frame

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

class Motion_Metric(IOU):
    def __init__(self):
        super().__init__(num_classes=3, ignore_cls=[-1])

        self.total_dynamic = 0
        self.total_static = 0
        self.pred_dynamic = 0
        self.pred_static = 0

    def update_coverage(self, nbr_static, nbr_dynamic, pred_stat, pred_dyn):

        # self.total_pts += nbr_pts
        self.total_dynamic += nbr_dynamic
        self.total_static += nbr_static
        self.pred_dynamic += pred_dyn
        self.pred_static += pred_stat

    def print_coverage(self):
        static_coverage = self.pred_static / self.total_static
        dynamic_coverage = self.pred_dynamic / self.total_dynamic
        total_coverage = (self.pred_static + self.pred_dynamic) / (self.total_static + self.total_dynamic)

        print(f"Total Coverage: {total_coverage * 100:.2f} % \t Static Coverage: {static_coverage * 100:.2f} \t Dynamic Coverage: {dynamic_coverage * 100:.2f} %")

    def reset(self):

        self.total_dynamic = 0
        self.total_static = 0
        self.pred_dynamic = 0
        self.pred_static = 0

        super().reset()

    def get_moving_stats(self):
        precs, recalls, ious = self._compute_stats(self.tps, self.fps, self.fns)

        return precs[2], recalls[2], ious[2]

def prior_metrics_sequence(sequence, Motion_metric=None, ego_prior=True, visibility_prior=False, static_prior=False, road_prior=True, per_frame_stats=False, store_mos=False, store_latex=False):

    covered_pts = 0
    all_pts_nbr = 0

    covered_dynamic_pts = 0
    all_dynamic_pts = 0

    for frame in range(len(sequence)):   # dynamic label flow, I need to adjust it

        prior_label = - np.ones(sequence.get_feature(idx=frame, name='lidar').shape[0], dtype=np.int32)
        dynamic_label = sequence.get_feature(idx=frame, name='dynamic_label')

        if ego_prior:
            ego_label = sequence.get_feature(idx=frame, name='corrected_ego_prior_label')
            # ego_label = sequence.get_feature(idx=frame, name='ego_prior_label')
            # ego_tm = sequence.get_feature(idx=frame, name='ego_tm')
            prior_label[ego_label == 1] = 1  # dynamic object
            prior_label[ego_label == 0] = 0  # road

        if visibility_prior:
            # visibility_label = sequence.get_feature(idx=frame, name='visibility_prior')
            visibility_label = sequence.get_feature(idx=frame, name='corrected_visibility_prior')
            # print(np.unique(visibility_label))
            prior_label[visibility_label > 0] = 1


        if static_prior:
            prior_static_mask = sequence.get_feature(idx=frame, name='prior_static_mask')
            prior_label[prior_static_mask==C.cfg['required_static_time']] = 0 # determistically static

        if road_prior:
            road_label = sequence.get_feature(idx=frame, name='road_proposal')
            prior_label[road_label == True] = 0

        # if store_mos:
            # sequence.store_feature(prior_label, frame, 'final_prior_label')
            # os.makedirs(sequence.sequence_path + '/' + 'final_prior_label_mos/', exist_ok=True)

            # store_mos_format(prior_label, sequence.sequence_path + '/' + 'final_prior_label_mos/' + f"{frame:06d}.label")

        # above is for ablation study

        # todo TMP!
        # prior_static_mask = sequence.get_feature(idx=frame, name='prior_static_mask')
        # prior_label[prior_static_mask == C.cfg['required_static_time']] = 0  # determistically static
        # prior_label[sequence.get_feature(frame, 'final_prior_label') == 1] = 1

        # add coverage and P and R check.
        covered_pts += np.sum(prior_label != -1)
        all_pts_nbr += prior_label.shape[0]

        covered_dynamic_pts += np.sum((prior_label == 1) & (dynamic_label == 1))
        all_dynamic_pts += np.sum(dynamic_label == 1)

        # valid_
        valid_mask = (prior_label != -1) & (dynamic_label != -1)    # valid chosen prior and without unlabelled ground truth

        pred_dyn = np.sum((dynamic_label == 1) & (prior_label == 1))
        total_dyn = np.sum((dynamic_label == 1))
        pred_stat = np.sum((dynamic_label == 0) & (prior_label == 0))
        total_stat = np.sum((dynamic_label == 0))
        # valid_mask = (dynamic_label != -1)    # valid chosen prior and without unlabelled ground truth
        # prior_label[valid_mask] = 0 # assign zero

        if prior_label.max() == -1:
            continue

        # breakpoint()

        Motion_metric.update(dynamic_label[valid_mask], prior_label[valid_mask])
        Motion_metric.update_coverage(total_stat, total_dyn, pred_stat, pred_dyn)

        if per_frame_stats:
            print(frame)
            Motion_metric.print_coverage()
            Motion_metric.print_stats()

    return Motion_metric


def foreground_sequence(sequence, accum_error_rate):

    stat_def = 70


    for frame in range(len(sequence)):   # dynamic label flow, I need to adjust it

        prior_label = sequence.get_feature(idx=frame, name='final_prior_label')
        # visibility
        static_prior = sequence.get_feature(idx=frame, name='prior_static_mask')



        dynamic_label = sequence.get_feature(idx=frame, name='dynamic_label')

        flow_label = sequence.get_feature(idx=frame, name='flow_label')
        lidarseg = sequence.get_feature(idx=frame, name='lidarseg')

        # pts = sequence.get_feature(idx=frame, name='lidar')

        foreground_label = - np.ones(static_prior.shape[0])
        foreground_label[(lidarseg > 0) & (lidarseg != 3)] = 1
        foreground_label[dynamic_label == 1] = 2

        # error cases - We say that moving object equals to static object
        # errors = (foreground_label == 2) & (static_prior >= stat_def)

        accum_error_rate['nbr_pts'] += np.sum(static_prior == stat_def)
        accum_error_rate['nbr_foreground_det'] += np.sum(prior_label == 1)
        accum_error_rate['nbr_gt_foreground'] += np.sum(dynamic_label == 1)

        accum_error_rate['nbr_errors'] += np.sum( (foreground_label == 2)[static_prior == stat_def])
        accum_error_rate['error_rate'] = accum_error_rate['nbr_errors'] / accum_error_rate['nbr_foreground_det']

        accum_error_rate['nbr_foreground_prior'] += np.sum(prior_label == 1)

    print(accum_error_rate)
        # print(f"Error_rate: {accum_error_rate['error_rate']:.2f}", 'nbr samples: ', errors[static_prior >= stat_def].shape)


        # waymo traffic - 3, unlabelled - -1, static background - 0
    # visualize_points3D(pts, prior_label != -1)
    # visualize_points3D(pts, errors)

    return accum_error_rate

def get_class_histogram(dataset):
    from my_datasets.kitti.semantic_kitti import kept_labels

    dataset = Waymo_Sequence
    # for waymo
    kept_labels = ['unlabelled', 'car', 'person', 'truck', 'bicycle', 'traffic sign']

    metric = IOU(len(kept_labels) + 1, clazz_names=kept_labels + ['garbage_idx'])
    garbage_idx = len(kept_labels)



    sequence = dataset(0)

    info_dict = sequence.info()

    for seq_id in range(0, info_dict['nbr_of_seqs']):
        # if seq_id > 9: continue
        sequence = dataset(sequence_nbr=seq_id)
        print(seq_id)
        for frame in range(len(sequence)):

            orig_seg_label = sequence.get_feature(idx=frame, name='lidarseg')
            seg_label = orig_seg_label.copy()
            my_label = sequence.get_feature(idx=frame, name='final_prior_label')


            seg_label[my_label == -1] = garbage_idx
            seg_label[seg_label <= 0] = garbage_idx
            orig_seg_label[orig_seg_label <= 0] = garbage_idx


            # metric.update(orig_seg_label, orig_seg_label)
            metric.update(orig_seg_label, seg_label)
        metric.print_stats()

    precs, recalls, ious = metric._compute_stats(metric.tps, metric.fps, metric.fns)

    # creating the dataset
    fig = plt.figure(figsize=(10, 6))

    # creating the bar plot
    plt.bar(list(kept_labels), list(recalls[:-1] * 100), color='maroon',
            width=0.4)

    plt.xticks(rotation=30, ha='right')
    plt.xlabel("Classes")
    plt.ylabel("Recall [%]")
    plt.title("Coverage of our dynamic / static estimation")
    plt.savefig('/home/vacekpa2/data/class_hist_waymo.png')


if __name__ == '__main__':
    IOU = Motion_Metric()


    from my_datasets.kitti.semantic_kitti import SemanticKitti_Sequence
    from my_datasets.waymo.waymo import Waymo_Sequence
    from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence

    import sys

    exp_dataframe = get_config_frames(PRIOR_METHODS, value)
    print(exp_dataframe)
    exp_nbr = int(sys.argv[1])

    exp = exp_dataframe.iloc[exp_nbr]
    print('Current experiment: \n', exp)

    # label_file = sequence.sequence_path + '/prior_mos_labels/' + str(frame).zfill(6) + '.label'
    # store_mos_format(ego_prior_label, label_file)

    # todo correct the moving labels in SemanticKitti for motion state

    for dataset in [Waymo_Sequence]: #  , ,

        # todo order datasequence and frames from highest dynamic coverage to lowest
        sequence = dataset(0)

        info_dict = sequence.info()
        accum_error_rate = {'nbr_pts': 0, 'nbr_foreground_prior': 0, 'nbr_errors': 0,
                            'nbr_foreground_det' : 0, 'nbr_gt_foreground' : 0, 'error_rate': 0}
        # print(info_dict)
        # get_class_histogram(dataset)
        for seq in range(0, info_dict['nbr_of_seqs']):

            # if seq > 1: continue #or seq == 27: continue

            print(dataset, seq)

            sequence = dataset(sequence_nbr=seq)
            # IOU = prior_metrics_sequence(sequence, IOU, ego_prior=exp[0], visibility_prior=exp[1], static_prior=exp[2], road_prior=exp[3], per_frame_stats=True, store_mos=True, store_latex=False)
            IOU = foreground_sequence(sequence, accum_error_rate=accum_error_rate)

            if seq > 30:
                break

            # IOU.print_stats()
            # IOU.print_coverage()

        # IOU.reset()
