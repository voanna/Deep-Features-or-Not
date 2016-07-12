#!/usr/bin/env python
from __future__ import print_function
from extractCaffeActivations import features
import argparse
import HONHelpers as hon
import itertools
import os
import glob


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

job_config_list = [pair for pair in itertools.product(hon.webcams, ['train', 'test'])]
 
# grid engine jobs start with 1
job_id = args.job_id - 1

job_config = job_config_list[job_id]
webcam, split = job_config
print(webcam, split)
finetune_root = os.path.join(hon.experiment_root, 'finetune-temperature', 'no-finetune-features')
img_fnames = sorted(glob.glob(os.path.join(hon.hon_data_root, webcam, 'imgs_align', '*' + split + '*.png')))
    
deploy = hon.VGG16_deploy_path
weights = hon.VGG16_caffemodel_path

layer = 'fc7'
save_directory = os.path.join(finetune_root, webcam) 
_ = features(deploy, weights, img_fnames, layer, save_directory, layers, mean_npy = None)





