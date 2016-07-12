import HONHelpers as hon
import extractCaffeActivations as eca
import os
import glob
import argparse
import itertools
import ipdb

ipdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument("job_id", help="indexes the job of extracting features", type=int)
args = parser.parse_args()
sequences = [
    '0000013443',
    '0000018918',
    '0000005705',
    '0000013029',
    '0000013207',
    '0000005394'
    ]

webcam = sequences[args.job_id -1 ]
caffe_deploy_prototxt = hon.VGG16_deploy_path
caffe_weights = hon.VGG16_caffemodel_path
save_directory = os.path.join(hon.experiment_root, 'time-prediction', webcam)
sequence_dir = hon.cvl_webcam_root
img_fnames = sorted(glob.glob(os.path.join(sequence_dir, webcam, '*.jpg')))
layer = 'pool4'
layers = ['pool4', 'fc6']
_ = eca.features(caffe_deploy_prototxt, caffe_weights, img_fnames, layer, save_directory, layers, mean_npy = None)