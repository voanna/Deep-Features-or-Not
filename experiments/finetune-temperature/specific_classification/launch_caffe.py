# 'source /home/voanna/TimePrediction/src/bash/gpu_caffe_env_variables ')
from __future__ import print_function
import os
import time_to_label
import glob
import math
import numpy as np
import scipy.io
import json
import argparse
import HONHelpers as hon
import random

parser = argparse.ArgumentParser()
parser.add_argument("webcam", help="either the name of the webcam you want to use from {} or 'all'".format(hon.webcams), type=str)
parser.add_argument("GPU_ID", help="gpu core to run the caffe training on", type=int)
args = parser.parse_args()


np.random.seed(6)
random.seed(6)
if args.webcam == 'all':
    webcams = hon.webcams
else:
    assert args.webcam in hon.webcams
    webcams = [args.webcam]

CAFFE_PATH = hon.gpu_caffe_root
CAFFE_MODEL = hon.VGG16_caffemodel_path
DATA_ROOT = hon.hon_data_root

EXPERIMENT_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.webcam)

training_frac = 0.8
MIN_TEMPERATURE = 5

with open(os.path.join(EXPERIMENT_ROOT, 'train.txt'), 'w') as ftrain, \
    open(os.path.join(EXPERIMENT_ROOT, 'val.txt'), 'w') as fval:

    for webcam in webcams:
        matfile = os.path.join(DATA_ROOT, webcam, 'train_data_aligned.mat')
        labels = scipy.io.loadmat(matfile)
        labels = labels['y']
        labels = labels[~np.isnan(labels)]
        labels = labels.astype(int) - MIN_TEMPERATURE

        img_path = os.path.join(DATA_ROOT, webcam, 'imgs_align')
        train_val_imgs = glob.glob(os.path.join(img_path, '*train*.png'))
        train_val_imgs = sorted(train_val_imgs)
        random.shuffle(train_val_imgs)
        # train on past, validate on future, test on even more future ???
        num_training = int(math.ceil(len(train_val_imgs)) * training_frac)
        train_imgs = train_val_imgs[:num_training]
        val_imgs = train_val_imgs[num_training:]
        train_labels = labels[:num_training]
        val_labels = labels[num_training:]
        assert train_imgs + val_imgs == train_val_imgs
        assert list(train_labels) + list(val_labels) == list(labels)

        for i in range(len(train_imgs)):
            ftrain.write(train_imgs[i] + ' ' + str(train_labels[i]) + '\n')    

        for i in range(len(val_imgs)):
            fval.write(val_imgs[i] + ' ' + str(val_labels[i]) + '\n')
       


snapshots = glob.glob(os.path.join(EXPERIMENT_ROOT, '*solverstate'))
if snapshots != []:
    idx = [int(snapshot[len(os.path.join(EXPERIMENT_ROOT, "model_iter_")):-len(".solverstate")]) for snapshot in snapshots]
    last = sorted(idx)[-1]
    os.system("{} train -solver {} -snapshot {} -gpu {} 2>&1 | tee --append {}".format(
        os.path.join(CAFFE_PATH, "build/tools/caffe"), 
        os.path.join(EXPERIMENT_ROOT, 'solver.prototxt'),
        os.path.join(EXPERIMENT_ROOT, "model_iter_" + str(last) + ".solverstate"),
        args.GPU_ID,
        os.path.join(EXPERIMENT_ROOT, "log.log")
        ))
else:
    os.system("{} train -solver {} -weights {} -gpu {} 2>&1 | tee {}".format(
        os.path.join(CAFFE_PATH, "build/tools/caffe"), 
        os.path.join(EXPERIMENT_ROOT, 'solver.prototxt'),
        CAFFE_MODEL,
        args.GPU_ID,
        os.path.join(EXPERIMENT_ROOT, "log.log")
        ))
