from __future__ import print_function
import extractCaffeActivations as eca
import os
import glob
import HONHelpers as hon
import numpy as np
import argparse
import itertools
import sys
import csv
import sklearn.metrics
import sklearn.neighbors
import sklearn.metrics
import time_to_label
import errno

import ipdb
np.random.seed(6)

def get_features(deploy, weights, train_img_fnames, test_img_fnames, layer, save_directory, layers, conf, mean_npy = None):
    """
    Extract or read from file features for network specified by deploy and weights, for a given layer, for images in train_ and test_img_fnames.

    Args:
        deploy: full path to a caffe model deploy prototxt specification, that can be used to test the network
        weights: full path to the .caffemodel weights matching the deploy
        train_img_fnames: a list of paths to training images
        test_img_fnames: a list to paths to testing images. 
        layer : which layer's activations to return, or a tuple of layers, in which case the indivual layers activations will be concatenated.
        save_directory : a directory which contains a subdirectory named after each layer.  
            Inside each layer's subdirectory, the activation for that layer for a particular image will be stored as image_basename_with_file_extension.txt
            If the features are not yet extracted, the layer subdirectories will be created inside the save_directory and features stored inside.
        layers : A list of all desired layers for the experiment. 
            To speed up the process of extracting features, we extract the activations of all desired layers simultaneously.
        conf: a Settings object for this experiment
        mean_npy : The mean image for a dataset to be used in preprocessing the image for caffe.  Leaving as None means using the ImageNet mean pixel value
    Returns:
        Two matrices, X_train and X_test, with dimensions len(img_fnames) x feature_dimensionality.  
        If layers is a tuple of layers, the feature for each image will be a simple concatenation of the features for each layer listed in the tuple.
        If the combined feature dimensionality is greater than 300'000, the features are projected to 500 dimensions.
        The PCA projection is learned on X_train and fitted to both X_train and X_test.

    """
    def reduce_dim(X_train, X_test):
        """
        Reduce the features in X_train, X_test to 500 dimensional features, learning a PCA transform on X_train

        Args:
            X_train: training features matrix, dims n_samples x feature_dimensionality
            X_test: testing features matrix
        Returns:
            PCA-projected versions of above.
        """
        import sklearn.decomposition
        pca = sklearn.decomposition.PCA(n_components = 500, whiten = True)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        return X_train, X_test

    if isinstance(layer, str):
        X_train = eca.features(deploy, weights, train_img_fnames, layer, save_directory, layers, mean_npy = None)
        X_test = eca.features(deploy, weights, test_img_fnames, layer, save_directory, layers, mean_npy = None)
    elif isinstance(layer, tuple):
        X_train = []
        X_test = []

        for l in layer:
            X_train.append(eca.features(deploy, weights, train_img_fnames, l, save_directory, layers, mean_npy = None))
            X_test.append(eca.features(deploy, weights, test_img_fnames, l, save_directory, layers, mean_npy = None))

        X_train = np.hstack(X_train)
        X_test = np.hstack(X_test)
    else:
        raise TypeError("'layer' must be either a string or tuple of strings, got {}".format(layer))

    if X_test.shape[1] > 300000:
        X_train, X_test = reduce_dim(X_train, X_test)

    return X_train, X_test


def svr_labels_no_finetune(train_img_fnames, test_img_fnames, y, layer, webcam, conf):
    """
    Use support vector regressor to predict label using features from layer activations

    Args:
        train_img_fnames: a list of paths to training images
        test_img_fnames: a list to paths to testing images.   
        y: labels for the training images
        layer : which layer's activations to use, or a tuple of layers, in which case the indivual layers activations will be concatenated and used as the feature
            Features are extracted from VGG-16 directly here.
        webcam: a webcam id
        conf: a Settings object with experimental parameters
    Returns:
        Predictions of labels for testing images from an SVR trained on training image features and labels.
    """
    deploy = hon.VGG16_deploy_path
    weights = hon.VGG16_caffemodel_path
    save_directory = os.path.join(conf.vgg_features_dir, webcam)    
    X, X_test = get_features(deploy, weights, train_img_fnames, test_img_fnames, layer, save_directory, conf.layers_to_extract_vgg, conf, mean_npy = None)    
    pred = hon.eval_svr(X, y, X_test, whitening = True)
    return pred

def svc_labels_no_finetune(train_img_fnames, test_img_fnames, y, layer, webcam, conf):
    """
    Use support vector machine (classifier) to predict label using features from layer activations

    Args:
        train_img_fnames: a list of paths to training images
        test_img_fnames: a list to paths to testing images.   
        y: labels for the training images
        layer : which layer's activations to use, or a tuple of layers, in which case the indivual layers activations will be concatenated and used as the feature
            Features are extracted from VGG-16 directly here.
        webcam: a webcam id
        conf: a Settings object with experimental parameters
    Returns:
        Predictions of labels for testing images from an SVM trained on training image features and labels.
    """
    deploy = hon.VGG16_deploy_path
    weights = hon.VGG16_caffemodel_path
    save_directory = os.path.join(conf.vgg_features_dir, webcam)    
    X, X_test = get_features(deploy, weights, train_img_fnames, test_img_fnames, layer, save_directory, conf.layers_to_extract_vgg, conf, mean_npy = None)    
    pred = hon.eval_svc(X, y, X_test, conf)
    return pred

def knn_labels(train_img_fnames, test_img_fnames, y, layer, webcam, conf):
    """
    Use K nearest neighbors classifier to predict label using features from layer activations. K is specified through the conf object.

    Args:
        train_img_fnames: a list of paths to training images
        test_img_fnames: a list to paths to testing images.   
        y: labels for the training images
        layer : which layer's activations to use, or a tuple of layers, in which case the indivual layers activations will be concatenated and used as the feature
            Features are extracted from VGG-16 directly here.
        webcam: a webcam id
        conf: a Settings object with experimental parameters
    Returns:
        Predictions of labels for testing images from a KNN clustering trained on training image features and labels.
    """
    deploy = hon.VGG16_deploy_path
    weights = hon.VGG16_caffemodel_path
    save_directory = os.path.join(conf.vgg_features_dir, webcam)    
    X, X_test = get_features(deploy, weights, train_img_fnames, test_img_fnames, layer, save_directory, conf.layers_to_extract_vgg, conf, mean_npy = None)    
    pred = hon.KNN(X, y, X_test, conf)
    return pred

def svr_labels_specific_finetune(train_img_fnames, test_img_fnames, y, layer, webcam, iteration, conf):
    """
    Use support vector regressor to predict label using features from layer activations

    Args:
        train_img_fnames: a list of paths to training images
        test_img_fnames: a list to paths to testing images.   
        y: labels for the training images
        layer : which layer's activations to use, or a tuple of layers, in which case the indivual layers activations will be concatenated and used as the feature
        iteration : The iteration of the finetuned model to use for extracting features.
        webcam: a webcam id to specify caffemodel, since this is the specific finetuning case.
        conf: a Settings object with experimental parameters
    Returns:
        Predictions of labels for testing images from an SVR trained on training image features and labels.
    """

    deploy = os.path.join(conf.HON_generic_webcams_expt_root, 'hon_vgg_deploy.prototxt')
    weights = os.path.join(conf.HON_specific_webcams_expt_root, webcam, 'model_iter_' + str(iteration) + '.caffemodel')
    save_directory = os.path.join(conf.HON_specific_webcams_expt_root, webcam)
    X, X_test = get_features(deploy, weights, train_img_fnames, test_img_fnames, layer, save_directory, conf.layers_to_extract_finetune, conf, mean_npy = None)    
    pred = hon.eval_svr(X, y, X_test, whitening = True)
    return pred


def svc_labels_AMOS_finetune(train_img_fnames, test_img_fnames, y, layer, finetune_mode, iteration, conf):

    deploy = os.path.join(conf.HON_specific_webcams_expt_root, finetune_mode, 'hon_vgg_deploy.prototxt')
    weights = os.path.join(conf.HON_specific_webcams_expt_root, finetune_mode, 'model_iter_' + str(iteration) + '.caffemodel')
    save_directory = os.path.join(conf.HON_specific_webcams_expt_root, finetune_mode, 'features_' + str(iteration))
    X, X_test = get_features(deploy, weights, train_img_fnames, test_img_fnames, layer, save_directory, [layer], conf, mean_npy = None)    
    pred = hon.eval_svc(X, y, X_test, conf, whitening = False)
    return pred

def direct_prob_prediction(test_img_fnames, finetune_mode, iteration, num_classes, conf):
    
    deploy = os.path.join(conf.HON_specific_webcams_expt_root, finetune_mode, 'hon_vgg_deploy.prototxt')
    weights = os.path.join(conf.HON_specific_webcams_expt_root, finetune_mode, 'model_iter_' + str(iteration) + '.caffemodel')
    save_directory = os.path.join(conf.HON_specific_webcams_expt_root, finetune_mode, 'features_' + str(iteration))
    X = eca.features(deploy, weights, test_img_fnames, 'prob', save_directory, ['prob'], mean_npy = None)
    pred = np.argmax(X, axis=1)
    # pred = [int(np.dot(X[i,:], np.arange(num_classes))) for i in range(X.shape[0])] 
    return pred

def svr_labels_generic_finetune(train_img_fnames, test_img_fnames, y, layer, webcam, iteration, conf):
    """
    Use support vector regressor to predict label using features from layer activations

    Args:
        train_img_fnames: a list of paths to training images
        test_img_fnames: a list to paths to testing images.   
        y: labels for the training images
        layer : which layer's activations to use, or a tuple of layers, in which case the indivual layers activations will be concatenated and used as the feature
        iteration : The iteration of the finetuned model to use for extracting features.
        webcam: a webcam id, needed to determine the save_directory in which to read / write features.
        conf: a Settings object with experimental parameters
    Returns:
        Predictions of labels for testing images from an SVR trained on training image features and labels.
    """

    deploy = os.path.join(conf.HON_generic_webcams_expt_root, 'hon_vgg_deploy.prototxt')
    if conf.expt == 'temperature-regression':
        weights = os.path.join(conf.HON_generic_webcams_expt_root, 'model_iter_' + str(iteration) + '.caffemodel')
        save_directory = os.path.join(conf.HON_generic_webcams_expt_root, 'HotOrNotVGG16', webcam)
    elif conf.expt == 'temperature-classification':
        weights = os.path.join(conf.HON_generic_webcams_expt_root, 'generic', 'model_iter_' + str(iteration) + '.caffemodel')
        save_directory = os.path.join(conf.HON_generic_webcams_expt_root,'generic', webcam)
    X, X_test = get_features(deploy, weights, train_img_fnames, test_img_fnames, layer, save_directory, conf.layers_to_extract_finetune, conf, mean_npy = None)    
    pred = hon.eval_svr(X, y, X_test, whitening = True)
    return pred

def record_job(index, webcam, layer, finetune_mode, target, pred, conf):
    """
    Saves true labels, predicted labels, as well as R squared and RMSE for one of the jobs determined by webcam, layer, and finetune_mode, as enumerated inside main()

    Args:
        index: index to the job_config_list of main, corresponding to the particular (webcam, layer, finetune_mode) combination
        webcam: webcam id
        layer: layer used to make predictions, can also be tuple of layers
        finetune_mode: type of finetuning used on model the features were extracted from
        target: ground truth labels for testing images
        pred: prediction from this job
        conf: Settings object
    Returns 
        Nothing
    """
    mods = {
    'season' :4,
    'month' : 12,
    'week' : 52,
    'day' : 365
    }

    logfile = os.path.join(conf.logdir, format(index, '04') + '.csv')
    predfile = os.path.join(conf.preddir, format(index, '04') + '.csv')
    targetfile = os.path.join(conf.targetdir, format(index, '04') + '.csv')
    if pred == None:
        row = '{}|"{}"|"{}"|"{}"|NaN|NaN\n'.format(index, webcam, layer, finetune_mode)
    else:
        np.savetxt(predfile, pred)
        np.savetxt(targetfile, target)
        try:
            mod = mods[finetune_mode]
        except KeyError:
            mod = None

        rsq_res = conf.acc(pred, target)
        rmse_res = conf.err(pred, target, mod = mod)
        row = '{}|"{}"|"{}"|"{}"|{:2.6f}|{:2.6f}\n'.format(index, webcam, layer, finetune_mode, rsq_res, rmse_res)
    print(row)
    with open(logfile, 'w') as f:
        f.write(row)


def get_averages(finetune_mode, layer, rsq_dict, rmse_dict, conf):
    """
    Average RMSE and R squared performance over webcam sequences

    Args:
        finetune_mode and layer : specify the setup for which to compute the average
        rsq_dict : A dictionary with keys being the job specification, i.e. (finetune_mode, layer, webcam) and values the R squared values
        rmse_dict : Same as above, but for RMSE values.
        conf : Settings object
    Returns:
        average R squared and RMSE values for specified setup
    """
    keys = [(finetune_mode, str(layer), webcam) for webcam in conf.webcams]
    rsq_av = np.mean([rsq_dict[key] for key in keys])
    rmse_av = np.mean([rmse_dict[key] for key in keys])
    return rsq_av, rmse_av

def read_csv_results(conf):
    """
    After all experiments have run, read in Rsquared and RMSE values for all jobs into dictionaries

    Args:
        conf: Settings object, containing fields where to find saved results, which are stored as CSVs
    Returns:
        Two dictionaries, for Rsq and RMSE, with entries for each job of the job_config_list, with the keys being (finetune_mode, str(layer), webcam)
    """
    rmse_dict = {}
    rsq_dict = {}
    fnames = sorted(glob.glob(os.path.join(conf.logdir, '*.csv')))
    for f in fnames:
        with open(f) as csvfile:
            reader = csv.reader(csvfile, delimiter = "|", quotechar = '"')
            for row in reader:
                rsq_dict[(row[3], row[2], row[1])] = np.float32(row[4])
                rmse_dict[(row[3], row[2], row[1])] = np.float32(row[5])
    return rsq_dict, rmse_dict

def temperature_classification_analysis(webcam, layer, finetune_mode, conf):
    """
    Function to choose correct setup for (webcam, layer, finetune_mode) combination that determines a job,
    for the temperature prediction experiment cast as a classification problem for the Neural Network.

    Args:
        webcam, layer, finetune_mode : these determine the job.
        conf: Settings object
    Return:
        True and predicted labels for testing images.  If the job does not make sense, sets predicted label to None.
    """
    #fc8HON never enters here
    assert conf.expt == 'temperature-classification'
    assert finetune_mode in ('no_finetune', 'generic_finetune', 'specific_finetune')

    # get training and testing images, and training and testing labels (all from Glasner)
    train_img_fnames = sorted(glob.glob(os.path.join(hon.hon_data_root, webcam, 'imgs_align', '*train*.png')))
    test_img_fnames = sorted(glob.glob(os.path.join(hon.hon_data_root, webcam, 'imgs_align', '*test*.png')))
    y = hon.get_labels(hon.hon_data_root, webcam, 'train')
    target = hon.get_labels(hon.hon_data_root, webcam, 'test')
    

    # select correct iteration of the finetuned model to use for getting features (neural activations)
    if finetune_mode == 'generic_finetune':
        iteration = conf.iteration_dict['generic']
    elif finetune_mode == 'specific_finetune':
        iteration =  conf.iteration_dict[webcam]

    if finetune_mode == 'no_finetune':
        pred = svr_labels_no_finetune(train_img_fnames, test_img_fnames, y, layer, webcam, conf)
    elif finetune_mode == 'generic_finetune':
        pred = svr_labels_generic_finetune(train_img_fnames, test_img_fnames, y, layer, webcam, iteration, conf)
    elif finetune_mode == 'specific_finetune':
        pred = svr_labels_specific_finetune(train_img_fnames, test_img_fnames, y, layer, webcam, iteration,conf)

    return target, pred

def temperature_regression_analysis(webcam, layer, finetune_mode, conf):
    """
    Function to choose correct setup for (webcam, layer, finetune_mode) combination that determines a job,
    for the temperature prediction experiment cast as a regression problem for the Neural Network.

    Args:
        webcam, layer, finetune_mode : these determine the job.
        conf: Settings object
    Return:
        True and predicted labels for testing images.  If the job does not make sense, sets predicted label to None.
    """
    assert conf.expt == 'temperature-regression'
    assert finetune_mode in ('no_finetune', 'generic_finetune', 'specific_finetune')

    iteration =  conf.iteration_dict[webcam]

    train_img_fnames = sorted(glob.glob(os.path.join(hon.hon_data_root, webcam, 'imgs_align', '*train*.png')))
    img_fnames = sorted(glob.glob(os.path.join(hon.hon_data_root, webcam, 'imgs_align', '*test*.png')))
    y = hon.get_labels(hon.hon_data_root, webcam, 'train')
    target = hon.get_labels(hon.hon_data_root, webcam, 'test')

    if finetune_mode == 'no_finetune':
        pred = svr_labels_no_finetune(train_img_fnames, img_fnames, y, layer, webcam, conf)
    elif finetune_mode == 'generic_finetune':
        pred = svr_labels_generic_finetune(train_img_fnames, img_fnames, y, layer, webcam, iteration, conf)
    elif finetune_mode == 'specific_finetune':
        if iteration == 0:
            pred = svr_labels_no_finetune(train_img_fnames, img_fnames, y, layer, webcam, conf)
        else:
            pred = svr_labels_specific_finetune(train_img_fnames, img_fnames, y, layer, webcam, iteration, conf)

    return target, pred

def time_analysis(webcam, layer, finetune_mode, conf):
    """
    Function to choose correct setup for (webcam, layer, finetune_mode) combination that determines a job,
    for the time prediction experiment cast as a classification problem for the Neural Network.

    Args:
        webcam, layer, finetune_mode : these determine the job.
        conf: Settings object
    Return:
        True and predicted labels for testing images.  If the job does not make sense, sets predicted label to None.
    """
    # get training and testing imagepaths
    with open(os.path.join(conf.vgg_features_dir, webcam, 'train.txt'), 'r') as fnames:
        train_img_fnames = [os.path.join(conf.image_dir, webcam, f.rstrip()) for f in fnames]
    with open(os.path.join(conf.vgg_features_dir, webcam, 'test.txt'), 'r') as fnames:
        img_fnames = [os.path.join(conf.image_dir, webcam, f.rstrip()) for f in fnames]
    y = np.loadtxt(os.path.join(conf.vgg_features_dir, webcam, finetune_mode + '_train.txt'))
    target = np.loadtxt(os.path.join(conf.vgg_features_dir, webcam, finetune_mode + '_test.txt'))
  
    if conf.classifier == 'knn':
        pred = knn_labels(train_img_fnames, img_fnames, y, layer, webcam, conf)
    elif conf.classifier == 'svc':
        pred = svc_labels_no_finetune(train_img_fnames, img_fnames, y, layer, webcam, conf)
    return target,  pred


# table 2
def print_finetuning_results_on_temperature(conf):
    for layer in conf.layers_svm:
        rsq_vals = []
        rmse_vals = []
        for expt in ['temperature-regression', 'temperature-classification']:
            conf = Setting(expt)
            rsq_dict, rmse_dict = read_csv_results(conf)
            for finetune_mode in ['generic_finetune', 'specific_finetune']:
                rsq_av, rmse_av = get_averages(finetune_mode, layer, rsq_dict, rmse_dict, conf)
                rsq_vals.append(rsq_av)
                rmse_vals.append(rmse_av)
        rsq_av, rmse_av = get_averages('no_finetune', layer, rsq_dict, rmse_dict, conf)
        rsq_vals.append(rsq_av)
        rmse_vals.append(rmse_av)
        line = "&".join(["{:2.2f} / {:2.2f}".format(max(0, rsq_vals[i]), rmse_vals[i]) for i in range(len(rsq_vals))])
        print('{} & {} \\\\'.format(layer, line))    

# table 7 & 8
def table_per_method(rsq_dict, rmse_dict, conf):
    letter_dict = {
    'a': '0000005394',
    'b': '0000005705',
    'c': '0000013029',
    'd': '0000013207',
    'e': '0000013443',
    'f': '0000018918'
    }
    keys = letter_dict.keys()
    layer = ('pool4', 'fc6')
    
    for finetune_mode in conf.finetune_modes:
        for l in layer:
            rsq_vals = np.zeros((len(conf.webcams)+1,))
            rmse_vals = np.zeros((len(conf.webcams)+1,))
            for i, letter in enumerate(sorted(keys)):
                webcam = letter_dict[letter]
                rsq_vals[i] = rsq_dict[(finetune_mode, l, webcam)]
                rmse_vals[i] = rmse_dict[(finetune_mode, l, webcam)]

            rsq_vals[-1] = np.mean(rsq_vals[:len(conf.webcams)])
            rmse_vals[-1] = np.mean(rmse_vals[:len(conf.webcams)])
            line = "&".join(["{:2.1f} / {:2.2f}".format(100*rsq_vals[i], rmse_vals[i]) for i in range(len(conf.webcams)+1)])
            if l == 'pool4':
                print('\multirow{{{}}}{{{}}}{{{}}} & {} & {} \\\\'.format('2','*', finetune_mode, l, line))
            else:
                print(' & {} & {} \\\\\\hline'.format(l, line))    

def table3(rsq_dict, rmse_dict):
    letter_dict = {
        'f' : '00000090' ,
        'b' : '00000156' ,
        'c' : '00000204' ,
        'g' : '00000338' ,
        'e' : '00000484' ,
        'h' : '00000842' ,
        'd' : '00004181' ,
        'i' : '00004556' ,
        'j' : '00015767' ,
        'a' : '00017603' 
    }
    keys = letter_dict.keys()
    layer = ('pool4', 'fc6')
    for l in layer:
        rsq_vals = np.zeros((len(hon.webcams)+1,))
        rmse_vals = np.zeros((len(hon.webcams)+1,))
        for i, letter in enumerate(sorted(keys)):
            webcam = letter_dict[letter]
            rsq_vals[i] = rsq_dict[('no_finetune', l, webcam)]
            rmse_vals[i] = rmse_dict[('no_finetune', l, webcam)]

        rsq_vals[-1] = np.mean(rsq_vals[:len(hon.webcams)])
        rmse_vals[-1] = np.mean(rmse_vals[:len(hon.webcams)])
        line = "&".join(["{:2.2f} / {:2.2f}".format(rsq_vals[i], rmse_vals[i]) for i in range(len(hon.webcams)+1)])
        print('\\textbf{{{}}} & {} \\\\\\hline'.format(l, line))

class Setting(object):
    """
    A class to hold information about the experiment like where the images are stored, attributes that are only meant to be read

    Fields:
        acc: function taking the predicted and true labels to calculate the accuracy
        err: function taking the predicted and true labels to calculate the error 
        vgg_features_dir: location of features from VGG16
        finetune_modes: type of finetuning that was done
        webcams: list of webcams used in the experiment
        HON_specific_webcams_expt_root: directory in which subdirectories webcam/layer/imagefilename.txt (feature) can be found, for specific finetuned features
        iteration_dict: dict with key being webcam name, and value being the lowest loss iteration
        layers_to_extract_finetune: layers to extract the features for, from the finetuned model
        layers_to_extract_vgg: layers to extract from VGG16
        layers_svm: layers and combinations of layers to use in the classifier
        analysis: switching between correct functions for a job 
        preddir: directory to save predicted labels for each job
        targetdir: directory to save true labels for each job
        HON_generic_webcams_expt_root: directory in which subdirectories webcam/layer/imagefilename.txt (feature) can be found, for generic finetuned features
        logdir: directory in which to save the Rsq and RMSE for each job
    """
    def __init__(self, expt, classifier=None):
        self.expt = expt
        self.HON_generic_webcams_expt_root = os.path.join(hon.experiment_root, self.expt)
        if expt == 'temperature-classification':
            self.acc = hon.rsq
            self.err = hon.rmse
            self.finetune_modes = ['no_finetune', 'generic_finetune', 'specific_finetune']
            self.webcams = hon.webcams
            self.vgg_features_dir = os.path.join(hon.experiment_root, 'finetune-temperature', 'no-finetune-features')
            self.HON_specific_webcams_expt_root = os.path.join(hon.experiment_root, 'finetune-temperature', 'specific-classification')
            self.HON_generic_webcams_expt_root = os.path.join(hon.experiment_root, 'finetune-temperature', 'generic-classification')

            self.iteration_dict = {
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

            self.layers_to_extract_finetune = [
                'pool1',
                'pool2',
                'pool3', 
                'pool4',  
                'pool5', 
                'fc6', 
                'fc7',
                ]

            self.layers_to_extract_vgg = [
                'pool1',
                'pool2',
                'pool3', 
                'pool4',  
                'pool5', 
                'fc6', 
                'fc7',
            ]

            self.layers_svm = [
                'pool1',
                'pool2',
                'pool3', 
                'pool4',  
                'pool5', 
                'fc6', 
                'fc7', 
                ('pool3', 'pool4'),
                ('fc6', 'fc7'),
            ]
            self.analysis = temperature_classification_analysis
        elif expt == 'temperature-regression':
            self.acc = hon.rsq
            self.err = hon.rmse
            self.finetune_modes = ['no_finetune', 'generic_finetune', 'specific_finetune']
            self.webcams = hon.webcams
            self.vgg_features_dir = os.path.join(hon.experiment_root, 'finetune-temperature', 'no-finetune-features')
            self.HON_specific_webcams_expt_root = os.path.join(hon.experiment_root, 'finetune-temperature', 'specific_classification')
            self.HON_generic_webcams_expt_root = os.path.join(hon.experiment_root, 'finetune-temperature', 'generic_classification')

            self.iteration_dict = {
                '00000090' : 10000,
                '00000156' : 10000,
                '00000204' : 10000,
                '00000338' : 10000,
                '00000484' : 10000,
                '00000842' : 10000,
                '00004181' : 10000,
                '00004556' : 10000,
                '00015767' : 10000,
                '00017603' : 10000,
                'generic'  : 10000
            }

            self.analysis = temperature_regression_analysis
            self.layers_to_extract_finetune = [
                'pool1',
                'pool2',
                'pool3', 
                'pool4',  
                'pool5', 
                'fc6', 
                'fc7',
                ]

            self.layers_to_extract_vgg = [
                'pool1',
                'pool2',
                'pool3', 
                'pool4',  
                'pool5', 
                'fc6', 
                'fc7',
            ]

            self.layers_svm = [
                'pool1',
                'pool2',
                'pool3', 
                'pool4',  
                'pool5', 
                'fc6', 
                'fc7', 
                ('pool3', 'pool4'),
                ('fc6', 'fc7'),
                ]
        elif expt == 'time-prediction':
            self.classifier = classifier
            self.acc = sklearn.metrics.accuracy_score
            self.err = hon.rmse
            self.layers_svm = ['pool4', 'fc6']
            self.layers_to_extract_vgg = self.layers_svm
            self.webcams = [   
                '0000013443',
                '0000018918',
                '0000005705',
                '0000013029',
                '0000013207',
                '0000005394'
            ]
            self.finetune_modes = ['season', 'month', 'week', 'day']
            self.vgg_features_dir = os.path.join(hon.experiment_root, "time-prediction")
            self.image_dir = hon.cvl_webcam_root
            self.analysis = time_analysis
            self.HON_specific_webcams_expt_root = os.path.join(hon.experiment_root, "time-prediction")
        elif expt == 'finetune-time':
            self.acc = sklearn.metrics.accuracy_score
            self.err = hon.rmse
            self.webcams = ['00017603']
            self.finetune_modes = [
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
            'vgg16_1_season',
            'vgg16_2_season'
            ]
            self.layers_to_extract_vgg = ['pool4', 'fc6']
            self.HON_specific_webcams_expt_root = self.HON_generic_webcams_expt_root
            self.analysis = finetuneAMOS
            self.layers_svm = ['pool4', 'fc6', 'prob']
            self.vgg_features_dir = os.path.join(self.HON_generic_webcams_expt_root, 'VGG-16', 'features')
        else:
            raise NotImplementedError

        if hasattr(self, 'classifier'):
            self.preddir = os.path.join(self.HON_specific_webcams_expt_root, 'pred', self.classifier)
            self.targetdir = os.path.join(self.HON_specific_webcams_expt_root, 'target', self.classifier)
            self.logdir = os.path.join(self.HON_specific_webcams_expt_root, 'temp', self.classifier)
        else:
            self.preddir = os.path.join(self.HON_specific_webcams_expt_root, 'pred')
            self.targetdir = os.path.join(self.HON_specific_webcams_expt_root, 'target')
            self.logdir = os.path.join(self.HON_specific_webcams_expt_root, 'temp')

        try_makedir(self.preddir)
        try_makedir(self.targetdir)
        try_makedir(self.logdir)
        

def try_makedir(folder_location):
    # pulled from http://stackoverflow.com/questions/1586648/race-condition-creating-folder-in-python
    try:
        os.makedirs(folder_location)
    except OSError, e:
        if e.errno == errno.EEXIST and os.path.isdir(folder_location):
            # File exists, and it's a directory,
            # another process beat us to creating this dir, that's OK.
            pass
        else:
            # Our target dir exists as a file, or different error,
            # reraise the error!
            raise

def finetuneAMOS(webcam, layer, finetune_mode, conf):
    def get_labels(img_list, year_div):
        if year_div == "daytime":
            labels = time_to_label.gen_labels(img_list, year_div, start=hon.start(AMOS_ID), end=hon.end(AMOS_ID))
        elif year_div == "hour":
            labels = time_to_label.gen_labels(img_list, year_div, start=hon.start(AMOS_ID))
        else:
            labels = time_to_label.gen_labels(img_list, year_div)
        return labels

    iteration_dict = {
        'vgg16_1_season' : None,
        'vgg16_2_season' : None,
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
        'vgg16_1_season' : ['pool4', 'fc6'],
        'vgg16_2_season' : ['pool4', 'fc6'],
        'year_1_day_1e-06_from_vgg16_1_frac'     : ['prob'],
        'year_1_daytime_1e-06_from_vgg16_1_frac' : ['prob'],
        'year_1_hour_1e-06_from_vgg16_1_frac'    : ['prob'],
        'year_1_month_1e-06_from_vgg16_1_frac'   : ['prob'],
        'year_1_season_1e-06_from_vgg16_1_frac'  : ['pool4', 'fc6', 'prob'],
        'year_1_week_1e-06_from_vgg16_1_frac'    : ['prob'],
        'year_2_day_1e-06_from_vgg16_1_frac'     : ['prob'],
        'year_2_daytime_1e-06_from_vgg16_1_frac' : ['prob'],
        'year_2_hour_1e-06_from_vgg16_1_frac'    : ['prob'],
        'year_2_month_1e-06_from_vgg16_1_frac'   : ['prob'],
        'year_2_season_1e-06_from_vgg16_1_frac'  : ['pool4','fc6', 'prob'],
        'year_2_week_1e-06_from_vgg16_1_frac'    : ['prob']
        }

    year_dict = {
        'vgg16_1_season' : [1, 4],
        'vgg16_2_season' : [2, 4],
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

    sequence_dir = os.path.join(hon.AMOS_root, '00017603')
    iteration = iteration_dict[finetune_mode]
    split = finetune_mode.split('_')
    year_div = split[2]
    if layer not in layers_dict[finetune_mode]:
        return None, None
    if layer == 'prob':
        img_fnames = hon.clean_image_list(sequence_dir)
        img_fnames = hon.slice_year(4, img_fnames)
        target = get_labels(img_fnames, year_div)
        num_classes = hon.time_divs[year_div]
        pred = direct_prob_prediction(img_fnames, finetune_mode, iteration, num_classes, conf)
    else:
        img_fnames = hon.clean_image_list(sequence_dir)
        train_img_fnames = hon.slice_year(year_dict[finetune_mode][0], img_fnames)
        test_img_fnames = hon.slice_year(4, img_fnames)
        y = get_labels(train_img_fnames, year_div)
        target = get_labels(test_img_fnames, year_div)
        if finetune_mode in ( 'vgg16_1_season',  'vgg16_2_season'):
            pred = svc_labels_no_finetune(
                train_img_fnames, test_img_fnames, y, layer, '', conf)
        else:
            pred = svc_labels_AMOS_finetune(
                train_img_fnames, test_img_fnames, y, layer, finetune_mode, iteration, conf)

    return target, pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("id", help="A number starting from 1, specifying job to run", type = int)
    parser.add_argument("experiment", help = "Can be either 'temperature-classification', 'temperature-regression' or 2016-03-08-biwi-webcams", type = str)
    parser.add_argument("--classifier", help = "Can be svc, svr or knn", type=str)
    parser.add_argument("--table", help = "Table id in paper, can be 2, 3, 7 or 8", type=int)
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    idx = args.id
    expt = args.experiment
    classifier = args.classifier  
    if classifier is not None:
        assert classifier in  ('svc', 'knn') #knn = 1NN
    return idx, expt, classifier, args.table

def main():
    """
    Compute how well a certain layer predicts the time or temperature for a given webcam, layer, finetuning combination.
    As this script is meant to run as batch script, one job gets processed per instance of running script
    """
    # index is the job id, indexing particular webcam, layer, finetune_mode combination
    # experiment is the name of the experiment as well as the directory name where the experimental results are stored
    index, expt, classifier, table = parse_args()
    conf = Setting(expt, classifier=classifier)
    if table is not None:
        if table == 2:
            conf = Setting('temperature-classification')
            print_finetuning_results_on_temperature(conf)
        elif table == 3:
            conf = Setting('temperature-classification')
            rsq_dict, rmse_dict = read_csv_results(conf)
            table3(rsq_dict, rmse_dict)
        elif table == 7:
            conf = Setting('time-prediction', classifier='svc')
            rsq_dict, rmse_dict = read_csv_results(conf)
            table_per_method(rsq_dict, rmse_dict, conf)
        elif table == 8:
            conf = Setting('time-prediction', classifier='knn')
            rsq_dict, rmse_dict = read_csv_results(conf)
            table_per_method(rsq_dict, rmse_dict, conf)
        else:
            # Tables 9 & 10 are created manually from the results, check README for details
            raise NotImplementedError
    else:
        job_config_list = [triple for triple in itertools.product(conf.webcams, conf.layers_svm, conf.finetune_modes)]
        assert 1 <= index <= len(job_config_list) 
        webcam, layer, finetune_mode = job_config_list[index - 1]
        print("job {} of {}".format(index, len(job_config_list)))
        print(webcam, layer, finetune_mode)
        # if job not yet computed, compute, else exit
        if not os.path.exists(os.path.join(conf.logdir, format(index, '04') + '.csv')):
            target, pred = conf.analysis(webcam, layer, finetune_mode, conf)
            record_job(index, webcam, layer, finetune_mode, target, pred, conf)
        else:
            print('Already computed')


if __name__ == '__main__':
    main()
