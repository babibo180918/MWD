import os
from os import walk
from os import listdir
from os.path import isfile, join
import cv2
import glob
import random
import matplotlib.pylab as plt
import numpy as np
import Utils
import config as cf


def image2input(filepaths, numOfClasses, label):
    numOfFiles = len(filepaths)
    images = []
    labels = np.array([], dtype=int)
    X = np.array([[]])
    
    # preprocessing image
    for f in range(0, numOfFiles):
        file = filepaths[f]
        image = cv2.imread(file, 0)
        if image is not None:
            images = np.append(images, file)
            labels = np.append(labels, label)
            # convert image to features
            resize_ = cv2.resize(image, (cf.WIDTH,cf.HEIGHT), interpolation=cv2.INTER_CUBIC)
            norm_ = cv2.normalize(resize_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            x = np.reshape(norm_,(norm_.shape[0]*norm_.shape[1], 1))
            if f == 0:
                X = x
            else:
                X = np.append(X, x, axis=1)

    datalen = len(images)
    y = Utils.one_hot_encode(numOfClasses, labels)
    
    return X, y, images

def create_dataset(in_path, out_path, numOfClasses, label, size):
    if label > numOfClasses:
        print('Error: label value must be less than number of classes.')
        return -1
    filepaths = []
    if os.path.isfile(in_path):
        filepaths.append(in_path)
    elif os.path.isdir(in_path):
        for filename in glob.glob(in_path + '/**/*.*', recursive=True):
            filepaths.append(filename)
    else:
        return -1
    
    # convert input image to model input.
    print("Num of files: " + str(len(filepaths)))
    X, y, images = image2input(filepaths, numOfClasses, label)
    datalen = len(images)
    print("Num of images: " + str(datalen))
    if size > datalen:
        size = datalen
    samples = random.sample(range(0,size), size)
    
    with open(out_path, 'wb') as out_file:
        np.savez(out_file, X=X, y=y, paths=images)
        print("Dataset has been saved to " + out_path)
    
    return 0

def load_dataset(in_path):
    filepaths = None
    X = None
    y = None
    with np.load(in_path) as data:
        X = data['X']
        y = data['y']
        filepaths = data['paths']    
    return X, y, filepaths
    
def add_dataset(in_path, dataset_path, numOfClasses, label):
    return 0
