import cv2
import os
from os import walk
from os import listdir
from os.path import isfile, join
import glob
import random
import matplotlib.pylab as plt
import numpy as np

def one_hot_encode(numOfClasses, labels):
    encoded_labels = np.zeros((numOfClasses, len(labels)), dtype=int)
    for i in range(0, len(labels)):
        if labels[i] > 0:
            encoded_labels[labels[i]-1, i] = 1
            
    return encoded_labels
    
