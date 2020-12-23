import os
import sys
from os.path import isfile
from datetime import date
import glob
import random
import matplotlib.pylab as plt
import numpy as np
import cv2
from skimage import io 
from skimage.transform import rotate, AffineTransform, warp
from skimage import img_as_ubyte
from skimage.util import random_noise

def anticlockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, -angle)

def h_flip(image):
    return  np.fliplr(image)

def v_flip(image):
    return np.flipud(image)

def add_noise(image):
    return random_noise(image)

def blur_image(image):
    return cv2.GaussianBlur(image, (9,9),0)

def warp_shift(image): 
    transform = AffineTransform(translation=(0,40))
    warp_image = warp(image, transform, mode="wrap")
    return warp_image

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
    num = 0
    filepaths = getFileList(in_path)
    for f in filepaths:
        image = cv2.imread(f, 0)
        if image is None:
           if os.path.exists(f):
               path, filename = os.path.split(f)
               filename, ext = os.path.splitext(filename)
               today = date.today()
               d = today.strftime("%Y%m%d")
               new_f = os.path.join(path, d + "_" + str(num) + ext)
               os.rename(f, new_f)
               image = cv2.imread(new_f, 0)
               if image is None:
                   os.remove(new_f)
                   print('Removed file: ' + new_f)
               else:
                   print('Renamed: ' + new_f)
               num = num + 1
    print('Done!')

def image_augmentation(in_path):
    '''
    duplicate images by different ways of image processing
    to augment training data of skewed dataset.(10X)
    '''
    
    transformations = {'rotate_anticlockwise': anticlockwise_rotation,
                       'rotate_clockwise': clockwise_rotation,
                       'horizontal_flip': h_flip, 
                       'vertical_flip': v_flip,
                       'warp_shift': warp_shift,
                       'adding_noise': add_noise,
                       'blurring_image':blur_image
                       }
    filepaths = getFileList(in_path)
    for f in filepaths:
        filename, ext = os.path.splitext(f)
        image = io.imread(f)
        n = 0
        while n < 10:
            key = random.choice(list(transformations)) #randomly choosing method to call
            transformed_image = transformations[key](image)
            new_image_path= '{}_{}_{}{}'.format(filename, key, n, ext)
            transformed_image = img_as_ubyte(transformed_image)  #Convert an image to unsigned byte format, with values in [0, 255].
            transformed_image=cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) #convert image to RGB before saving it
            cv2.imwrite(new_image_path, transformed_image) # save transformed image to path
            n = n + 1
    
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
    
    
