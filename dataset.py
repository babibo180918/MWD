import os
from os.path import isfile
from pathlib import Path
import sys
import glob
import random
import matplotlib.pylab as plt
import numpy as np
import fex
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
        x = fex.fex(file)
        if x is not None:
            images = np.append(images, file)
            labels = np.append(labels, label)
            if len(images) == 1:
                X = x
            else:
                X = np.append(X, x, axis=1)
        if len(images)%100 == 0:
            print('Extracted {} images ...'.format(len(images)), end='\r')        

    print('{} image(s) have been dispatched.'.format(len(images)))
    # create labels
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
    numOfMiniBatch = len(filepaths)//cf.MINIBATCH_SIZE
    for i in range(0,numOfMiniBatch):
        X, y, images = image2input(filepaths[i*cf.MINIBATCH_SIZE:(i+1)*cf.MINIBATCH_SIZE], numOfClasses, label)
        with open(out_path, 'ab') as out_file:
            batch = {'X':X, 'y':y, 'paths':images}
            np.save(out_file, batch)
            print("{} samples has been added to {}".format(len(images), out_path))
        del X,y,images
    with open(out_path, 'ab') as out_file:
        X, y, images = image2input(filepaths[numOfMiniBatch*cf.MINIBATCH_SIZE:len(filepaths)], numOfClasses, label)
        batch = {'X':X, 'y':y, 'paths':images}
        np.save(out_file, batch)
        print("{} samples has been added to {}".format(len(images), out_path))
    return 0

def load_dataset(in_path):
    filepaths = None
    X = None
    y = None
    p = Path(in_path)
    file_sz = os.stat(in_path).st_size
    i = 0
    with p.open('rb') as f:
        while f.tell() < file_sz:
            batch = np.load(f, allow_pickle=True)
            i = i+1
                
def add_dataset(in_path, dataset_path, numOfClasses, label):
    return 0
