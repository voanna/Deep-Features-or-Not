# 'source /home/voanna/TimePrediction/src/bash/gpu_caffe_env_variables ')

import os
import time_to_label
import glob
import math
import numpy as np
import scipy.io

np.random.seed(6)

CAFFE_PATH = '/home/voanna/caffe_gpu'
CAFFE_MODEL = 'VGG_ILSVRC_16_layers.caffemodel'
GPU_ID = 0
DATA_ROOT = os.path.expanduser('~/TimePrediction/data/hot_or_not/data')
EXPERIMENT_ROOT = os.path.expanduser('~/TimePrediction/experiments/2016-02-04-hon-finetune/')
webcams = ['00000090',  '00000156',  '00000204',  '00000338',
  '00000484',  '00000842',  '00004181',  '00004556',
  '00015767',  '00017603']

training_frac = 0.8

with open(os.path.join(EXPERIMENT_ROOT, 'train.txt'), 'w') as ftrain, \
    open(os.path.join(EXPERIMENT_ROOT, 'val.txt'), 'w') as fval:

    for webcam in webcams:
        matfile = os.path.join(DATA_ROOT, webcam, 'train_data_aligned.mat')
        labels = scipy.io.loadmat(matfile)
        labels = labels['y']
        labels = labels[~np.isnan(labels)]

        img_path = os.path.join(DATA_ROOT, webcam, 'imgs_align')
        train_val_imgs = glob.glob(os.path.join(img_path, '*train*.png'))
        train_val_imgs = sorted(train_val_imgs)

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
        GPU_ID,
        os.path.join(EXPERIMENT_ROOT, "log.log")
        ))
else:
    os.system("{} train -solver {} -weights {} -gpu {} 2>&1 | tee {}".format(
        os.path.join(CAFFE_PATH, "build/tools/caffe"), 
        os.path.join(EXPERIMENT_ROOT, 'solver.prototxt'),
        os.path.join(EXPERIMENT_ROOT, CAFFE_MODEL),
        GPU_ID,
        os.path.join(EXPERIMENT_ROOT, "log.log")
        ))
