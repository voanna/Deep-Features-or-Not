import os
import glob
import scipy.io as io
import numpy as np
import sklearn
import socket
import sklearn.svm
import sys
import matplotlib.pyplot as plt
import pylab
import mpl_toolkits.axes_grid1

project_root = os.path.expanduser('~/TimePrediction/src/public/')

AMOS_root = os.path.join(project_root, 'data', 'AMOS')
VGG16_caffemodel_path = os.path.join(project_root, 'VGG16','VGG_ILSVRC_16_layers.caffemodel')
VGG16_deploy_path = os.path.join(project_root, 'VGG16','VGG_ILSVRC_16_layers_deploy.prototxt')
hon_data_root = os.path.join(project_root, 'data','hot_or_not','data')
experiment_root =  os.path.join(project_root, 'experiments')
gpu_caffe_root = '/home/voanna/caffe_gpu'
cvl_webcam_root = os.path.join(project_root(project_root, 'data', 'CVL_cams')

time_divs = {
    'season' : 4,
    'month' : 12,
    'week' : 52,
    'day': 365,
    'daytime' : 4,
    'hour' : 10
}

time_labels = {
    'season' : ['winter', 'spring', 'summer', 'fall'],
    'month' : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'week' : ['' if i%5 else str(i) for i in range(52)],
    'day': ['' if i%50 else str(i) for i in range(365)]
}

svm_opts = {
    'nu': 0.5,
    'C': 100,
    'kernel': 'linear',
    'shrinking': False,
    'tol': 0.001,
    'cache_size': 200,
    'verbose': True,
    'max_iter': -1
    }

svc_opts = {
    'kernel' : 'linear',
    'verbose' : True
}
webcams = [
        '00000090',
        '00000156',
        '00000204',
        '00000338',
        '00000484',
        '00000842',
        '00004181',
        '00004556',
        '00015767',
        '00017603'
    ]

def rmse(pred_labels, true_labels, units = 'C', mod = None):
    '''
    returns root mean square error as from Glasner paper, eq(2)

    >>> rmse(np.array([0,0,0,0]), np.array([1,2,3,4]))
    2.7386127875258306
    '''
    def F2C(F):
        return (5.0 / 9.0) * (F - 32.0)

    assert units in ('F', 'C')
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    if units == 'C':
        pred_labels = F2C(pred_labels)
        true_labels = F2C(true_labels)
    diff = np.abs(pred_labels - true_labels)
    if mod == None:
        return np.sqrt(np.mean(diff**2))
    else:
        return np.sqrt(np.mean((np.minimum(diff, mod - diff)**2)))

def rsq(pred_labels, true_labels):
    ''' 
    Returns coefficient of detemination (R^2), as in (1) of Glasner paper

    >>> rsq(np.array([0,0,0,0]), np.array([1,2,3,4]))
    -5.0
    '''
    num = np.sum((pred_labels - true_labels)**2)
    den = np.sum((np.mean(true_labels) - true_labels)**2)
    return 1 - (num/den)

def get_labels(data_root, webcam, split):
    '''
    loads matfile labels into ndarray
    '''
    assert split in ("train", "test")

    matfile = os.path.join(os.path.expanduser(data_root), webcam, split + '_data_aligned.mat')
    labels = io.loadmat(matfile)
    labels = labels['y']
    labels = labels[~np.isnan(labels)]
    return labels

def eval_svr(X, y, X_test, whitening = False, normalize = True, svm_opts_dict = svm_opts):

    regressor = sklearn.svm.NuSVR(**svm_opts_dict)

    if normalize:
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        X_test = min_max_scaler.fit_transform(X_test)
   


    if whitening:
        meanY = np.mean(y)
        stdY = np.std(y)
        y = y - meanY
        y = y / stdY
    
    regressor.fit(X, y)
    pred_labels = regressor.predict(X_test)
    if whitening:
        pred_labels = pred_labels * stdY
        pred_labels = pred_labels + meanY

    return pred_labels

def KNN(X, y, X_test, conf, normalize = True):
    if normalize:
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        X_test = min_max_scaler.fit_transform(X_test)
   
    neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)

    neigh.fit(X, y) 
    pred_labels = neigh.predict(X_test)
    return pred_labels


def eval_svc(X, y, X_test, conf, whitening = False, normalize = True, svm_opts_dict = svm_opts):
    num_classes = len(np.unique(y))

    svc = sklearn.svm.SVC(**svc_opts)

    if normalize:
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        X_test = min_max_scaler.fit_transform(X_test)

    svc.fit(X, y)

    pred_labels = svc.predict(X_test)

    return pred_labels


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

        segment = [i for i in img_list if start_date <= i < end_date]

    return segment

def clean_image_list(sequence_dir):
    assert sequence_dir == os.path.join(AMOS_root, '00017603')

    images = glob.glob(os.path.join(sequence_dir, "*jpg"))
    images = [os.path.basename(f) for f in images]
    images = sorted(images)

    # magic downsampling, determined graphically
    # after this, sampling should be mostly uniform
    oversampled = images[14448:62461]
    downsampled = oversampled[::10]
    uniform = images[:14448] + downsampled + images[62461:]
    return uniform

def start(AMOS_ID):
    if AMOS_ID == '00017603':
        return 13
    else:
        raise NotImplementedError

def end(AMOS_ID):
    if AMOS_ID == '00017603':
        return 23    
    else:
        raise NotImplementedError