import os
import sys
import os.path
from pathlib import Path
import numpy as np
import random
import tf2_CNN_config as cf
import dataset as ds
import Utils
import tensorflow as tf
import cv2
import pydot
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import *

# Model
cost_function="binary_crossentropy"
optimizer="SGD"
initializer="random"
mwd_cnn_model_output="./MODEL/MWD_CNN.json"
mwd_cnn_weights_output="./MODEL/MWD_CNN.h5"

model_path = mwd_cnn_model_output
weights_path = mwd_cnn_weights_output

# Dataset
train_dataset_path = cf.out_mixed_balanced_dataset_train
test_dataset_path = cf.out_mixed_balanced_dataset_test

# Parameters
BATCH_SIZE = 128
learning_rate = 0.01
num_epochs = 100
L2_lambd = 0.0
beta1 = 0.0
beta2 = 0.0
epsilon = 0.0
print_cost = True

def dataset_steps(filepaths, bs):
    steps = len(filepaths) // bs
    return steps

def data_generator(filepaths, labels, bs):
    datasize = len(filepaths)
    print("data size: " + str(datasize))
    if bs > datasize:
        bs = datasize
    batch_X = np.zeros((bs, cf.WIDTH, cf.HEIGHT, 3))
    batch_Y = np.zeros((bs, 1))
    while True:
        i = 0
        for f in range(0,datasize):
            image = cv2.imread(filepaths[f])
            if image is not None:
                resize_ = cv2.resize(image, (cf.WIDTH,cf.HEIGHT), interpolation=cv2.INTER_CUBIC)
                #norm_ = cv2.normalize(resize_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                batch_X[i,:,:,:] = np.divide(resize_, 255.0, dtype=np.float32) - 0.5
                batch_Y[i,0] = labels[f]
                i = i + 1
                if i == bs:
                    i = 0
                    yield(batch_X, batch_Y)

def data_generator2(dataset, bs):
    p = Path(dataset)
    file_sz = os.stat(dataset).st_size
    f = p.open('rb')
    while True:
        if(f.tell()>= file_sz):
            f.seek(0)
        sector = np.load(f, allow_pickle=True)
        sector_X = sector[()]['X'].T # (numOfSamples, numOfFeatures)
        sector_Y = sector[()]['y'].T        
        batchPerSector = sector_X.shape[0]//bs
        sector_X = sector_X.reshape((sector_X.shape[0], cf.WIDTH, cf.HEIGHT, 1))
        for m in range(0,batchPerSector):
            start = m*bs
            end = (m+1)*bs
            batch_X = sector_X[start:end]
            batch_Y = sector_Y[start:end]
            yield(batch_X, batch_Y)
    

def MWD_CNN(input_shape, classes, initializer="random"):
    '''
    default value of number of labels is 1
    '''
    
    X_input = Input(input_shape) # 64x64x3
    X = ZeroPadding2D((2,2))(X_input) # 68x68x3
    X = Conv2D(64, (5,5), strides=(1,1), name="conv0", activation='relu')(X) # 64x64x64
    X = MaxPooling2D((2,2), name="max_pool0")(X) # 32x32x64
    X = Conv2D(128, (3,3), strides=(1,1), name="conv1", activation='relu')(X) #30x30x128
    X = AveragePooling2D((2,2), name="average_pool0")(X) # 15x15x128
    X = Flatten()(X)
    X = Dense(16, activation='relu', name='FC0')(X)
    X = Dense(classes, activation='sigmoid', name='FC1')(X)
    
    model = Model(inputs = X_input, outputs = X, name='MWD_CNN')
    return model
    
def train(model, filepaths, labels, batch_size, num_epochs, learning_rate, beta1, beta2, epsilon, model_path, print_cost=False):
    steps_per_epoch = dataset_steps(filepaths, batch_size)
    if optimizer == "SGD":
        opt = optimizers.SGD(learning_rate = learning_rate)
    elif optimizer == "adam":
        opt = optimizers.Adam(learning_rate = learning_rate)
    else:
        opt = optimizers.SGD(learning_rate = learning_rate)
        
    model.compile(optimizer=opt, loss=cost_function, metrics=['accuracy',Precision(), Recall()])
    model.fit(data_generator(filepaths, labels, batch_size), steps_per_epoch=steps_per_epoch, epochs=num_epochs)
    
def predict(model, filepaths, labels):
    datasize = len(filepaths)
    X = []
    for f in range(0,datasize):
        image = cv2.imread(f)
        if image is not None:
            resize_ = cv2.resize(image, (cf.WIDTH,cf.HEIGHT), interpolation=cv2.INTER_CUBIC)
            norm_ = np.divide(resize_, 255.0, dtype=np.float32) - 0.5
            X.appends(norm_)
    preds = model.evaluate(x=X, y=labels)
    print ("Loss = " + str(preds[0]))
    print ("Accuracy = " + str(preds[1]))
    print ("Precision = " + str(preds[2]))
    print ("Recall = " + str(preds[3]))

# filtering out the non-image files
#Utils.image_filter(cf.IN_MASKED_FACE_PATH)
#Utils.image_filter(cf.IN_FACE_PATH)

# mixed balanced dataset
masked_filepaths = Utils.getFileList(cf.IN_MASKED_FACE_PATH)
masked_labels = [1]*len(masked_filepaths) 
# suffle random masked images
sample_idxs = random.sample(range(0,len(masked_filepaths)), len(masked_filepaths))
masked_filepaths = [masked_filepaths[i] for i in sample_idxs]
masked_labels = [masked_labels[i] for i in sample_idxs]

face_filepaths = Utils.getFileList(cf.IN_FACE_PATH)
face_labels = [0]*len(face_filepaths)
# get 30000 samples of faces
sample_idxs = random.sample(range(0,len(face_filepaths)), 30000)       
face_filepaths = [face_filepaths[i] for i in sample_idxs]
face_labels = [face_labels[i] for i in sample_idxs]

filepaths = []
labels = []
filepaths.extend(masked_filepaths)
filepaths.extend(face_filepaths)
labels.extend(masked_labels)
labels.extend(face_labels)
sumOfSamples = len(labels)
sample_idxs = random.sample(range(0,sumOfSamples), sumOfSamples)         
filepaths = [filepaths[i] for i in sample_idxs]
labels = [labels[i] for i in sample_idxs]

# Model
model = MWD_CNN((cf.WIDTH, cf.HEIGHT, 3),cf.NUM_OF_CLASSES, initializer)
train(model, filepaths[0:30080], labels[0:30080], BATCH_SIZE, num_epochs, learning_rate, beta1, beta2, epsilon, model_path, print_cost)
predict(model, filepaths[30080:len(filepaths)], labels[30080:len(labels)])

