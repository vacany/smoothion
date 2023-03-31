import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from datatools.stats.segmentation import IOU
from datatools.visualizer.vis import *
from motion_segmentation.prepare_data import Motion_Sequence

sequence = 1
metric = IOU(2)

dataset = Motion_Sequence(sequence, start=300, end=450)
data = dataset.get_raw_data()

pcls = np.concatenate(data['pcls'])

removert = dataset.get_specific_data('/removert/dynamic/*', form='concat')
dynamic_labels = np.concatenate(data['dynamic_labels'])

RADIUS_range = [0.5, 1, 1.5, 2., 2.5, 3.5, 4.]
EPS_range = [0.5, 0.8, 1., 1.5, 2., 2.5]

for radius in RADIUS_range:
    metric.reset()

    ego_radius = dataset.get_specific_data(f'/ego_radius/radius_{radius:.1f}/*.npy', form='concat')
    print("RADIUS ", radius )
    metric.update(dynamic_labels, ego_radius)
    metric.print_stats()


metric.reset()
print("REMOVERT")
metric.update(dynamic_labels, removert)
metric.print_stats()

metric.reset()
print("BOTH")
metric.update(dynamic_labels, (removert + ego_radius) > 0)
metric.print_stats()


