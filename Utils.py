import os
from os.path import isfile
import glob
import random
import matplotlib.pylab as plt
import numpy as np
import cv2

def one_hot_encode(numOfClasses, labels):
    encoded_labels = np.zeros((numOfClasses, len(labels)), dtype=int)
    for i in range(0, len(labels)):
        if labels[i] > 0:
            encoded_labels[labels[i]-1, i] = 1
            
    return encoded_labels

def getFileList(in_path):
    filepaths = []
    if os.path.isfile(in_path):
        filepaths.append(in_path)
    elif os.path.isdir(in_path):
        for filename in glob.glob(in_path + '/**/*.*', recursive=True):
            filepaths.append(filename)
    else:
        print("Path is invalid: " + in_path)
        return None

    return filepaths

def image_filter(in_path):
    filepaths = getFileList(in_path)
    for f in filepaths:
        image = cv2.imread(f, 0)
        if image is None:
           if os.path.exists(f):
              os.remove(f)
              print('Removed file: ' + f)
    print('Done!')
    
