import os
from os.path import isfile
from pathlib import Path
import sys

import random
import matplotlib.pylab as plt
import numpy as np

import fex
import Utils
import config as cf


def image2input(filepaths, numOfClasses, label):
    numOfFiles = len(filepaths)
    images = []
    update_labels = np.array([], dtype=int)
    X = np.array([[]])
    
    # preprocessing image
    for f in range(0, numOfFiles):
        file = filepaths[f]
        x = fex.fex(file)
        if x is not None:
            images = np.append(images, file)
            update_labels = np.append(update_labels, label[f])
            if len(images) == 1:
                X = x
            else:
                X = np.append(X, x, axis=1)
        else:
            print("File error: " + file)
        if len(images)%100 == 0:
            print('Extracted {} images ...'.format(len(images)), end='\r')        

    print('{} image(s) have been dispatched.'.format(len(images)))
    # create labels
    y = Utils.one_hot_encode(numOfClasses, update_labels)
    
    return X, y, images

def create_dataset(in_path, out_path, numOfClasses, label, size):
    if label > numOfClasses:
        print('Error: label value must be less than number of classes.')
        return -1
    filepaths = Utils.getFileList(in_path)
    if filepaths is None:
        return -1
    
    # convert input image to model input.
    print("Num of files: " + str(len(filepaths)))
    numOfMiniBatch = len(filepaths)//cf.MINIBATCH_SIZE
    for i in range(0,numOfMiniBatch):
        labels = [label]*cf.MINIBATCH_SIZE
        X, y, images = image2input(filepaths[i*cf.MINIBATCH_SIZE:(i+1)*cf.MINIBATCH_SIZE], numOfClasses, labels)
        with open(out_path, 'ab') as out_file:
            batch = {'X':X, 'y':y, 'paths':images}
            np.save(out_file, batch)
            print("{} samples has been added to {}".format(len(images), out_path))
        del X,y,images,labels
    with open(out_path, 'ab') as out_file:
        labels = [label]*(len(filepaths)-cf.MINIBATCH_SIZE*numOfMiniBatch)
        X, y, images = image2input(filepaths[numOfMiniBatch*cf.MINIBATCH_SIZE:len(filepaths)], numOfClasses, labels)
        batch = {'X':X, 'y':y, 'paths':images}
        np.save(out_file, batch)
        print("{} samples has been added to {}".format(len(images), out_path))
    return 0

def create_mixed_dataset(out_path, numOfClasses, *args):
    filepaths = []
    labels = []    
    numOfArgs = len(args)
    
    if numOfArgs%2 != 0:
        print("Number of arguments are incorrect.")
        return

    for i in range(0, numOfArgs, 2):
        filepaths.extend(args[i])
        labels.extend(args[i+1])

    numOfDataset = numOfArgs//2
    sumOfSamples = len(labels)
    sample_idxs = random.sample(range(0,sumOfSamples), sumOfSamples)         
    filepaths = [filepaths[i] for i in sample_idxs]
    labels = [labels[i] for i in sample_idxs]

    # convert input image to model input.
    print("Num of files: " + str(len(filepaths)))
    numOfMiniBatch = len(filepaths)//cf.MINIBATCH_SIZE
    for i in range(0,numOfMiniBatch):
        X, y, images = image2input(filepaths[i*cf.MINIBATCH_SIZE:(i+1)*cf.MINIBATCH_SIZE], numOfClasses, labels[i*cf.MINIBATCH_SIZE:(i+1)*cf.MINIBATCH_SIZE])
        with open(out_path, 'ab') as out_file:
            batch = {'X':X, 'y':y, 'paths':images}
            np.save(out_file, batch)
            print("{} samples has been added to {}".format(len(images), out_path))
        del X,y,images
    with open(out_path, 'ab') as out_file:
        X, y, images = image2input(filepaths[numOfMiniBatch*cf.MINIBATCH_SIZE:len(filepaths)], numOfClasses, labels)
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

    with p.open('rb') as f:
        while f.tell() < file_sz:
            batch = np.load(f, allow_pickle=True)
            print(batch[()]['X'])
            print(batch[()]['y'])
            print(batch[()]['paths'])
            break;
                
def add_dataset(in_path, dataset_path, numOfClasses, label):
    return 0
