import os
import sys
import os.path
from pathlib import Path
import numpy as np
import random
import tensorflow as tf
import cv2
import pydot
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.metrics import *

from mwd import tf2_CNN_config
from mwd.model import mwd

DATASET_PATH = os.path.abspath("./data/mixed_balanced_dataset.npy")
OUTPUT_PATH = "./output"
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
# Model
cost_function="binary_crossentropy"
optimizer="SGD"
initializer="random"
mwd_cnn_model_output="./output/MWD_CNN.json"
mwd_cnn_weights_output="./output/MWD_CNN.h5"

model_path = mwd_cnn_model_output
weights_path = mwd_cnn_weights_output

# Parameters
BATCH_SIZE = 64
learning_rate = 0.01
num_epochs = 2
L2_lambd = 0.0
beta1 = 0.0
beta2 = 0.0
epsilon = 0.0
print_cost = True

def create_dataset(out_path, *args):
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
    inputs = {'filepaths':filepaths, 'labels':labels}
    np.save(out_path, inputs)
    return (filepaths, labels)

def dataset_steps(filepaths, bs):
    steps = len(filepaths) // bs
    return steps

def data_generator(filepaths, labels, bs):
    datasize = len(filepaths)
    print("data size: " + str(datasize))
    if bs > datasize:
        bs = datasize
    numOfBatch = datasize//bs
    print("numOfBatch: " + str(numOfBatch))
    batch_X = np.zeros((bs, tf2_CNN_config.WIDTH, tf2_CNN_config.HEIGHT, 3))
    batch_Y = np.zeros((bs, 1))
    while True:
        i = 0
        for i in range(0,numOfBatch):
            filepaths_batch = filepaths[i*bs:(i+1)*bs]
            batch_X = np.array([cv2.resize(cv2.imread(file_name), (tf2_CNN_config.WIDTH, tf2_CNN_config.HEIGHT)) for file_name in filepaths_batch])/255.0 - 0.5
            batch_Y = np.array(labels[i*bs:(i+1)*bs])
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
        sector_X = sector_X.reshape((sector_X.shape[0], tf2_CNN_config.WIDTH, tf2_CNN_config.HEIGHT, 1))
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
        image = cv2.imread(filepaths[f])
        if image is not None:
            resize_ = cv2.resize(image, (tf2_CNN_config.WIDTH,tf2_CNN_config.HEIGHT), interpolation=cv2.INTER_CUBIC)
            norm_ = np.divide(resize_, 255.0, dtype=np.float32) - 0.5
            X.append(norm_)
    preds = model.evaluate(x=np.array(X), y=np.array(labels))
    print ("Loss = " + str(preds[0]))
    print ("Accuracy = " + str(preds[1]))
    print ("Precision = " + str(preds[2]))
    print ("Recall = " + str(preds[3]))

#############################
#   Image pre-processing    #
#############################

# filtering out the non-image files
#Utils.image_filter(cf.IN_MASKED_FACE_AUGMENTED_PATH)
#Utils.image_filter(cf.IN_FACE_PATH)

# dataset augmentation
#Utils.image_augmentation(cf.IN_MASKED_FACE_AUGMENTED_PATH)

'''
# create mixed balanced dataset
masked_filepaths = Utils.getFileList(cf.IN_MASKED_FACE_AUGMENTED_PATH)
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
filepaths, labels = create_dataset(cf.out_mixed_balanced_dataset, masked_filepaths, masked_labels, face_filepaths, face_labels)
'''

# load dataset
inputs = np.load(DATASET_PATH, allow_pickle=True)
filepaths = inputs.item().get('filepaths')
labels = inputs.item().get('labels')
filepaths_train, filepaths_val, labels_train, labels_val = train_test_split(filepaths, labels, test_size=0.95, random_state=1)

# Model
#model = MWD_CNN((tf2_CNN_config.WIDTH, tf2_CNN_config.HEIGHT, 3),tf2_CNN_config.NUM_OF_CLASSES, initializer)
model = mwd.MWD_Model(Input((tf2_CNN_config.WIDTH, tf2_CNN_config.HEIGHT, 3)),classes=tf2_CNN_config.NUM_OF_CLASSES, name='MWD_Model')

train(model, filepaths_train, labels_train, BATCH_SIZE, num_epochs, learning_rate, beta1, beta2, epsilon, model_path, print_cost)
model.summary()

# serialize model to JSON
model_json = model.to_json()
with open(mwd_cnn_model_output, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(mwd_cnn_weights_output)
print("Saved model to disk")
'''
inputs = np.load(DATASET_PATH, allow_pickle=True)
filepaths = inputs.item().get('filepaths')
labels = inputs.item().get('labels')
filepaths_train, filepaths_val, labels_train, labels_val = train_test_split(filepaths, labels, test_size=0.2, random_state=1)

json_file = open(mwd_cnn_model_output, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.compile(optimizer=optimizers.SGD(learning_rate = learning_rate), loss=cost_function, metrics=['accuracy',Precision(), Recall()])
model.load_weights(mwd_cnn_weights_output)
'''

predict(model, filepaths_val, labels_val)
