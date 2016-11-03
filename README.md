# README
This repo contains the code used to run the experiments in ["Deep Features or Not: Temperature and Time Prediction in Outdoor Scenes"](http://www.vision.ee.ethz.ch/~timofter/publications/Volokitin-CVPRW-2016.pdf) by Anna Volokitin, Radu Timofte and Luc Van Gool, published in the CVPR 2016 Workshop on Robust Features.

We provide instructions on how to reproduce the tables in the paper

Part of the experiments rely on data and code provided by Glasner et. al.
Command-line caffe is a dependency as well

voanna AT vision.ee.ethz.ch

# Directory Structure
## Experiments
```
experiments/finetune-temperature
experiments/time-prediction
experiments/finetune-time
```

## Utility functions
`src/HONHelpers.py` is the utility module.  The paths should be set there first

## VGG16 caffemodel weights and prototxt
`VGG16/`

To get the source code and the caffemodels that the scripts use, first download the compressed caffemodels using
```
wget http://data.vision.ee.ethz.ch/voanna/TimePrediction/caffemodels.tgz
```
which already have the correct directory structure.

Then clone the sources from github into the directory containing the large files
(http://stackoverflow.com/questions/2411031/how-do-i-clone-into-a-non-empty-directory)

## Data
For data, create a subdirectory called data, and place datasets in following directories:

### Hot or Not Temperature Dataset
`data/hot_or_not/data`
Download data from Glasner et al. :
```
wget http://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/HotOrNot/resources/data/data.tar.gz
```

### Time of Year Dataset (ours)
`data/CVL_cams`
Our dataset, download :
```
wget http://data.vision.ee.ethz.ch/voanna/TimePrediction/data/CVL_cams.tgz
```

### Time of Year Webcam from AMOS (used for the final finetuning experiment, 4.4 years worth of images)
`data/AMOS/00017603` 
Data downloaded from AMOS database (http://amos.cse.wustl.edu/) but cleaned :
```
wget http://data.vision.ee.ethz.ch/voanna/TimePrediction/data/AMOS/00017603.tgz
```

# Table 2
`experiments/finetune-temperature/`

For this experiment, we first finetune VGG-16, extract features and then compute the performance of these features.  Each of these steps is explained below.

All the output files for this experiment will be placed in the directory `experiments/finetune-temperature/`

The directories
*	`specific_regression/`
*	`generic_regression/`
*	`specific_classification/`
*	`generic_classification/`

contain finetuned caffemodels at the iteration as reported in our paper.

To extract the features from these models, we use the script `extract_features_finetune_temperature.py`.

To extract unfinetuned features,  we use `extract_features_no_finetune_temperature.py` which is located in the experiment directory.

To compute the R^2/RMSE, run 
```
python finetuning_hon_performance.py JOB_ID temperature-classification
```
or
```
python finetuning_hon_performance.py JOB_ID temperature-classification
```
both of which include comparison with non-finetuned features.

This script is written to be run on a grid-compute system as there are many SVMs to train, so a second argument is a job id (an index into the SVMs to train).  The results of the SVM predictions, and the accuracies will be recorded in individual files, as configured in the Settings object in finetuning_hon_performance.py

After all the jobs are done, the individual results will be assembled into a table when running with 
```
python finetuning_hon_performance.py 1 temperature-classification svr --table 2
```

# Table 3
In table 3, we use the same results as computed in Table 2, so just run 
```
python finetuning_hon_performance.py 1 temperature-classification svm --table 3
```

# Table 7
`experiments/time-prediction/`

Extract features using `extract_features_time_prediction.py`.
To compute the R^2/RMSE, results are computed with run finetuning_hon_performance.py with 
```
python finetuning_hon_performance.py #{all ids} time-prediction 
```

To print the table to screen:
```
python finetuning_hon_performance.py #{all ids} time-prediction --classifier svc --table 7
```

# Table 8
When using 1-NN as a classifier, we still use the same features as above, so we can go directly to evaluating

To print the table to screen:
```
python finetuning_hon_performance.py #{all ids} time-prediction --classifier knn --table 8
```

# Table 9 & 10	
These belong to the same experiment, finetune-time
For extracting features, run `extract_features_finetune_time.py`

To compute the R^2/RMSE, run finetuning_hon_performance.py with 'finetune-time'
```
python finetuning_hon_performance.py #{all ids} time-prediction 
```

These tables were not generated programmatically, but the numbers were just taken from the files saved to the logdir , i.e. `os.path.join(hon.experiment_root, 'finetune-time', 'temp')`.
Running `cat *` in that directory gives a result like the following.

```
$ cat *
1|"00017603"|"pool4"|"year_1_day_1e-06_from_vgg16_1_frac"|NaN|NaN
2|"00017603"|"pool4"|"year_1_daytime_1e-06_from_vgg16_1_frac"|NaN|NaN
3|"00017603"|"pool4"|"year_1_hour_1e-06_from_vgg16_1_frac"|NaN|NaN
4|"00017603"|"pool4"|"year_1_month_1e-06_from_vgg16_1_frac"|NaN|NaN
5|"00017603"|"pool4"|"year_1_season_1e-06_from_vgg16_1_frac"|0.384368|0.701175
....
```
