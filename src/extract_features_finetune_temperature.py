#!/usr/bin/env python
from __future__ import print_function
from extractCaffeActivations import features
import argparse
import HONHelpers as hon
import itertools
import os
import glob

iteration_dict = {
        '00000090' : 100000,
        '00000156' : 0,
        '00000204' : 6000,
        '00000338' : 16000,
        '00000484' : 2000,
        '00000842' : 10000,
        '00004181' : 6000,
        '00004556' : 100000,
        '00015767' : 0,
        '00017603' : 2000,
        'generic'  : 100000 
    }

layers = [
    'pool1',
    'pool2',
    'pool3', 
    'pool4',  
    'pool5', 
    'fc6', 
    'fc7', 
    ]

parser = argparse.ArgumentParser()
parser.add_argument("job_id", help="indexes the job of extracting features", type=int)
args = parser.parse_args()

job_config_list = [pair for pair in itertools.product(hon.webcams, ['train', 'test'], ['specific', 'generic'], ['classification', 'regression'])]

# grid engine jobs start with 1
job_id = args.job_id - 1

job_config = job_config_list[job_id]
webcam, split, finetuning_type, setup = job_config
print(webcam, split, finetuning_type, setup)
experiment_root = os.path.join(hon.experiment_root, 'finetune-temperature')
finetune_root = os.path.join(experiment_root, finetuning_type + '_' + setup)
img_fnames = sorted(glob.glob(os.path.join(hon.hon_data_root, webcam, 'imgs_align', '*' + split + '*.png')))
    
deploy = os.path.join(finetune_root, 'hon_vgg_deploy.prototxt')
if setup == 'regression':
    if finetuning_type == 'specific':
        weights = os.path.join(finetune_root, webcam, 'model_iter_' + str(10000) + '.caffemodel')
    elif finetuning_type == 'generic':
        weights = os.path.join(finetune_root, 'model_iter_' + str(10000) + '.caffemodel')
else:
    if iteration_dict[webcam] == 0:
        weights = hon.VGG16_caffemodel_path
    else:
        if finetuning_type == 'specific':
            weights = os.path.join(finetune_root, webcam, 'model_iter_' + str(iteration_dict[webcam]) + '.caffemodel')
        elif finetuning_type == 'generic':
            weights = os.path.join(finetune_root, 'model_iter_' + str(iteration_dict['generic']) + '.caffemodel')

layer = 'fc7'
save_directory = os.path.join(finetune_root, webcam) 
_ = features(deploy, weights, img_fnames, layer, save_directory, layers, mean_npy = None)





