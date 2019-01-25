#from easydict import EasyDict as edict
import numpy as np



# Consumers can get config by:
#   from config import cfg

#anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
#anchors = np.array([[17, 28], [41, 57], [61, 159],[102, 89],[125, 251],[196, 147],[224, 331],[348, 217],[377, 383]])
anchors = np.array([[29, 76], [38, 42], [50, 49],[59, 62],[66, 25],[75, 72],[86,44],[86, 88],[112, 108]])
num = 9
num_anchors_per_layer = 3
batch_size = 2
scratch = False
names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#
# Training options
#

ignore_thresh = .5
momentum = 0.9
decay = 0.0005
learning_rate = 0.0001
max_batches = 50200
lr_steps = [40000, 45000]
lr_scales = [.1, .1]
max_truth = 100
mask = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
image_resized = 416   # { 320, 352, ... , 608} multiples of 32

#
# image process options
#
angle = 0
saturation = 1.5
exposure = 1.5
hue = .1
jitter = .3
random = 1
