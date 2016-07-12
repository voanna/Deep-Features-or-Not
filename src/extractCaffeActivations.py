# source environment variables here ? cpu caffe
import os
import sys
import glob
import numpy as np
import itertools
import pylab

# # Make sure that caffe is on the python path:
# caffe_root = '/home/voanna/caffe_cpu/'  # this file is expected to be in {caffe_root}/examples
# import sys 
# sys.path.insert(0, caffe_root + 'python')

def init(caffe_deploy_prototxt, caffe_weights, mean_npy, image_dims=[256, 256]):
    '''
    Sets up deploy version of the caffe model in python according to caffe_deploy_prototxt, 
    and loads the pretrained/finetuned weights from caffe_weights (a .caffemodel file)
    mean_npy is the mean image over the dataset.
    Assumes model expects a 0-255 image scaling
    '''
    import caffe

    net = caffe.Classifier(caffe_deploy_prototxt, caffe_weights, image_dims = image_dims)
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    net.transformer.set_mean('data', np.load(mean_npy).mean(1).mean(1)) # mean pixel
    net.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    return net

def retrieve_features(caffe_deploy_prototxt, caffe_weights, mean_npy, img_fnames, layer, save_directory, layers):
    '''
    Takes images in img_fnames, gets the neural activations of each image
    in layer, and returns a numpy matrix, saving the individual features along the way
    '''
    def read_df(out_file):
        try:
            with open(out_file,'r') as f:
                df = f.read()
                assert df != '', 'empty file is {}'.format(out_file)
                df = [float(x) for x in df.split()]
                df = np.atleast_1d(df)
        except IOError as e:
            raise e
        return df

    featnames = [img + '.txt' for img in img_fnames]
    all_outfiles = [os.path.join(*triple) for triple in itertools.product([save_directory], layers, [os.path.basename(img) + '.txt' for img in img_fnames])]
    all_computed = all([os.path.exists(outfile) for outfile in all_outfiles])
    if not all_computed:
        net = init(caffe_deploy_prototxt, caffe_weights, mean_npy)

    assert isinstance(img_fnames, list)
    print("Getting {} layer features".format(layer))    

    if all_computed:
        print('done')
        name = os.path.basename(img_fnames[0])
        out_file = os.path.join(save_directory, layer, name + '.txt')
        f1 = read_df(out_file)
    else:
        f1 = np.atleast_1d(getDeepFeatureLayers(net, img_fnames[0], [layer])[0])

    dim = f1.shape[0]
    features = np.zeros((len(img_fnames), dim), dtype = np.float64, order = 'C')

    for i, fname in enumerate(img_fnames):
        name = os.path.basename(fname)
        out_file = os.path.join(save_directory, layer, name + '.txt')
        try:
           df = read_df(out_file)
        except IOError:
            write_features(net, img_fnames, layers, save_directory)
            df = read_df(out_file)
        
        if df.shape[0] == 1:
            features[i] = df
        else:
            features[i, :] = np.asarray(df)

    return features


def getDeepFeatureLayers(net, imgPath, layers):
    import caffe
    image = caffe.io.load_image(imgPath)
    scores = net.predict([image]) 

    def accessBlob(net, layer):
        deep_feature = net.blobs[layer].data
        deep_feature = deep_feature.squeeze()
        deep_feature = deep_feature.mean(0)

        if len(deep_feature.shape)>1:
            deep_feature=np.hstack(np.hstack(deep_feature))
        return deep_feature

    df = [accessBlob(net, layer) for layer in layers]        
    return df



def write_features(net, img_fnames, layers, save_directory):
    assert isinstance(img_fnames, list)
    assert isinstance(layers, list)
    for layer in layers:
        layer_directory = os.path.join(save_directory, layer)
        if not os.path.isdir(layer_directory):
            os.makedirs(layer_directory)

    print("extracting")
    n = len(img_fnames)
    for i, fname in enumerate(img_fnames):
        name = os.path.basename(fname)
        compute = False
        for layer in layers:
            out_file = os.path.join(save_directory, layer, name + '.txt')
            if not os.path.isfile(out_file):
                compute = True # in case any of the features is not computed, we just extract them all
                break
        if compute:
            print(save_directory)
            print("{}, {} of {}".format(name, i, n))
            dflist = getDeepFeatureLayers(net, fname, layers)
            for df, layer in zip(dflist, layers):
                df = np.atleast_1d(df)
                out_file = os.path.join(save_directory, layer, name + '.txt')
                if not os.path.isfile(out_file):
                    with open(out_file,'w') as f:
                        for idx in range(df.shape[0]):
                            f.write('%5.10f ' % df[idx])


def features(caffe_deploy_prototxt, caffe_weights, 
    img_fnames, layer, save_directory, layers, mean_npy = None, gpu_id=None):
    if gpu_id != None:
        import caffe
        caffe.set_device(gpu_id)
    if mean_npy is None:
        mean_npy = '/home/voanna/caffe_cpu/python/caffe/imagenet/ilsvrc_2012_mean.npy'
        print('Using imagenet mean pixel')
    else:
        print('Using custom mean')
    feats = retrieve_features(caffe_deploy_prototxt, caffe_weights, mean_npy, img_fnames, layer, save_directory, layers)
    # net = init(caffe_deploy_prototxt, caffe_weights, mean_npy)
    # feats = retrieve_features(net, img_fnames, layer, save_directory, layers)
    return feats

def main():
    return

if __name__ == '__main__':
    main()
