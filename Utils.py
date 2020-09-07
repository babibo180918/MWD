import os
import sys
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
    
def skewed_error_analysis(Y, Y_pred, filepaths, printpath = False):
    true_pos = ((Y==1)&(Y_pred==1))
    false_pos = ((Y==0)&(Y_pred==1))
    true_neg = ((Y==0)&(Y_pred==0))
    false_neg = ((Y==1)&(Y_pred==0))
    
    true_pos_count = np.count_nonzero(true_pos == True, axis=1, keepdims=True)
    false_pos_count = np.count_nonzero(false_pos == True, axis=1, keepdims=True)
    true_neg_count = np.count_nonzero(true_neg == True, axis=1, keepdims=True)
    false_neg_count = np.count_nonzero(false_neg == True, axis=1, keepdims=True)
    #
    accuracy = (true_pos_count + true_neg_count)/Y.shape[1]
    precision = true_pos_count/(true_pos_count + false_pos_count)
    recall = true_pos_count/(true_pos_count + false_neg_count)
    f1_score = 2*precision*recall/(precision + recall)

    np.set_printoptions(threshold=sys.maxsize)
    print('True positive: {}'.format(true_pos_count))
    print('False positive: {}'.format(false_pos_count))
    if printpath:
        print(str(filepaths[false_pos[0]]))
    print('True negative: {}'.format(true_neg_count))
    print('False negative: {}'.format(false_neg_count))
    if printpath:
        print(str(filepaths[false_neg[0]]))
        
    print('Accuracy: {}'.format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1_score: {}'.format(f1_score))
    
    
