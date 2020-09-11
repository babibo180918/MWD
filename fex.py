import os
import cv2
import numpy as np
import config as cf

def fex(filepath):
    x = None
    image = cv2.imread(filepath, 0)
    if image is not None:
        resize_ = cv2.resize(image, (cf.WIDTH,cf.HEIGHT), interpolation=cv2.INTER_CUBIC)
        #norm_ = cv2.normalize(resize_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_ = resize_/255.0 - 0.5
        x = np.reshape(norm_,(norm_.shape[0]*norm_.shape[1], 1))        
    return x

