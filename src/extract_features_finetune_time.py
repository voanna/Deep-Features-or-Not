from __future__ import print_function
from extractCaffeActivations import features
import argparse
import HONHelpers as hon
import itertools
import os
import glob
import ipdb

def slice_year(year, img_list):
    """ 
    If year is 1 or 2, takes the first year or first two years respectively.
    If year is 3, takes only third year
    Assumes img_list contains only datestrings which can be sorted.
    Format is YYYYMMDD_HHSSmm
    If year == 4, take everything from start of 4th year to the end.
    """

    start_date = list(img_list[0])
    if year in (3, 4):
        start_year = int("".join(start_date[0:4])) + 1*(year - 1)
        start_date[0:4] = list(str(start_year))
    start_date = "".join(start_date)

    if year == 4:
        segment = [i for i in img_list if start_date <= i]
    else:
        duration = 1    
        if year == 2:
            duration = 2
        end_date = list(start_date)
        next_year = int(start_date[0:4]) + duration
        end_date[0:4] = list(str(next_year))
        end_date = "".join(end_date)

        print(start_date)
        print(end_date)
        segment = [i for i in img_list if start_date <= i < end_date]

    return segment

def clean_image_list(sequence_dir):
    images = glob.glob(os.path.join(sequence_dir, "*jpg"))
    images = [os.path.basename(f) for f in images]
    images = sorted(images)

    # magic downsampling, determined graphically
    # after this, sampling should be mostly uniform
    oversampled = images[14448:62461]
    downsampled = oversampled[::10]
    uniform = images[:14448] + downsampled + images[62461:]
    return uniform

experiments = [
    'year_1_day_1e-06_from_vgg16_1_frac' ,
    'year_1_daytime_1e-06_from_vgg16_1_frac' ,
    'year_1_hour_1e-06_from_vgg16_1_frac' ,
    'year_1_month_1e-06_from_vgg16_1_frac' ,
    'year_1_season_1e-06_from_vgg16_1_frac' ,
    'year_1_week_1e-06_from_vgg16_1_frac' ,
    'year_2_day_1e-06_from_vgg16_1_frac' ,
    'year_2_daytime_1e-06_from_vgg16_1_frac' ,
    'year_2_hour_1e-06_from_vgg16_1_frac' ,
    'year_2_month_1e-06_from_vgg16_1_frac' ,
    'year_2_season_1e-06_from_vgg16_1_frac' ,
    'year_2_week_1e-06_from_vgg16_1_frac' ,
    ]

iteration_dict = {
    'year_1_day_1e-06_from_vgg16_1_frac'     : 4000,
    'year_1_daytime_1e-06_from_vgg16_1_frac' : 2000,
    'year_1_hour_1e-06_from_vgg16_1_frac'    : 6000,
    'year_1_month_1e-06_from_vgg16_1_frac'   : 2000,
    'year_1_season_1e-06_from_vgg16_1_frac'  : 4000,
    'year_1_week_1e-06_from_vgg16_1_frac'    : 2000,
    'year_2_day_1e-06_from_vgg16_1_frac'     : 2000,
    'year_2_daytime_1e-06_from_vgg16_1_frac' : 18000, 
    'year_2_hour_1e-06_from_vgg16_1_frac'    : 18000,
    'year_2_month_1e-06_from_vgg16_1_frac'   : 2000,
    'year_2_season_1e-06_from_vgg16_1_frac'  : 6000,
    'year_2_week_1e-06_from_vgg16_1_frac'    : 2000
    }


layers_dict = {
    'year_1_day_1e-06_from_vgg16_1_frac'     : ['prob'],
    'year_1_daytime_1e-06_from_vgg16_1_frac' : ['prob'],
    'year_1_hour_1e-06_from_vgg16_1_frac'    : ['prob'],
    'year_1_month_1e-06_from_vgg16_1_frac'   : ['prob'],
    'year_1_season_1e-06_from_vgg16_1_frac'  : ['pool4','fc6', 'prob'],
    'year_1_week_1e-06_from_vgg16_1_frac'    : ['prob'],
    'year_2_day_1e-06_from_vgg16_1_frac'     : ['prob'],
    'year_2_daytime_1e-06_from_vgg16_1_frac' : ['prob'],
    'year_2_hour_1e-06_from_vgg16_1_frac'    : ['prob'],
    'year_2_month_1e-06_from_vgg16_1_frac'   : ['prob'],
    'year_2_season_1e-06_from_vgg16_1_frac'  : ['pool4','fc6', 'prob'],
    'year_2_week_1e-06_from_vgg16_1_frac'    : ['prob']
    }

year_dict = {
    'year_1_day_1e-06_from_vgg16_1_frac'     : [4],
    'year_1_daytime_1e-06_from_vgg16_1_frac' : [4],
    'year_1_hour_1e-06_from_vgg16_1_frac'    : [4],
    'year_1_month_1e-06_from_vgg16_1_frac'   : [4],
    'year_1_season_1e-06_from_vgg16_1_frac'  : [1, 4],
    'year_1_week_1e-06_from_vgg16_1_frac'    : [4],
    'year_2_day_1e-06_from_vgg16_1_frac'     : [4],
    'year_2_daytime_1e-06_from_vgg16_1_frac' : [4],
    'year_2_hour_1e-06_from_vgg16_1_frac'    : [4],
    'year_2_month_1e-06_from_vgg16_1_frac'   : [4],
    'year_2_season_1e-06_from_vgg16_1_frac'  : [2, 4],
    'year_2_week_1e-06_from_vgg16_1_frac'    : [4]
    }


AMOS_ID = '00017603'
EXPT_ROOT = os.path.join(hon.experiment_root, 'finetune-time')
n = 100

parser = argparse.ArgumentParser()
parser.add_argument("job_id", help = "A number between 0 and {}, inclusive".format(n), type=int)
args = parser.parse_args()

job_list = [pair for pair in itertools.product(experiments, range(10))]
assert 1<= args.job_id <= n
job_id = args.job_id - 1
experiment, chunk_id = job_list[job_id]

print(experiment)
print("Job {} of {}".format(job_id + 1, len(experiments) * n))
finetune_root = os.path.join(EXPT_ROOT, experiment)
deploy = os.path.join(finetune_root, 'hon_vgg_deploy.prototxt')
weights = os.path.join(finetune_root, 'model_iter_' + str(iteration_dict[experiment]) + '.caffemodel')
sequence_dir = os.path.join(hon.AMOS_root, AMOS_ID)
save_directory = os.path.join(finetune_root, 'features_' + str(iteration_dict[experiment]))
if not os.path.isdir(save_directory):
    os.makedirs(save_directory)
images = clean_image_list(sequence_dir)
layers = layers_dict[experiment]
for years in year_dict[experiment]:
    img_fnames = slice_year(years, images)
    img_fnames = [os.path.join(sequence_dir, f) for f in img_fnames]
    chunks = [img_fnames[i::n] for i in range(n)]
    for imgs in chunks:
        _ = features(deploy, weights, chunks[chunk_id], layers[0], save_directory, layers, mean_npy=None)




