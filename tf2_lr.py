import os
import sys
import os.path
from pathlib import Path
import numpy as np
import config as cf
import dataset as ds
import Utils
import tensorflow as tf
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
initializer="zero"
zero_init_model_output="./MODEL/tf_lr_zero_init.npy"
model_path = zero_init_model_output

# Dataset
train_dataset_path = cf.out_mixed_balanced_dataset_train
test_dataset_path = cf.out_mixed_balanced_dataset_test

# Parameters
MINIBATCH_SIZE = 100
learning_rate = 0.1
num_epochs = 10000
L2_lambd = 0.0
beta1 = 0.0
beta2 = 0.0
epsilon = 0.0
print_cost = True

def dataset_steps(dataset, bs):
    p = Path(dataset)
    file_sz = os.stat(dataset).st_size
    f = p.open('rb')
    steps = 0
    while f.tell() < file_sz:
        sector = np.load(f, allow_pickle=True)
        sector_X = sector[()]['X'].T # (numOfSamples, numOfFeatures)
        steps = steps + sector_X.shape[0]//bs
    print("batch size: " + str(bs))
    print("steps of dataset: " + str(steps))
    return steps

def data_generator(dataset, bs):
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
    

def MWD_lr(size, cost_function="cross_entropy", optimizer="GradientDescent", initializer="random", L2_regularizer=0.01):
    '''
    default value of number of labels is 1
    '''
    X_input = Input((cf.WIDTH, cf.HEIGHT, 1))
    X = Flatten()(X_input)
    X = Dense(1, 
        activation='sigmoid', name='logistic_regression',
        kernel_regularizer=regularizers.l2(l2=L2_regularizer),
        kernel_initializer=initializers.Zeros(), bias_initializer=initializers.Zeros()
    )(X)
    model = Model(inputs=X_input, outputs=X, name="MWD_lr")
    return model

def train(model, X_train, Y_train, minibatch_size, num_epochs, learning_rate, beta1, beta2, epsilon, model_path, print_cost=False):
    if optimizer == "SGD":
        opt = optimizers.SGD(learning_rate = learning_rate)
    elif optimizer == "adam":
        opt = optimizers.Adam(learning_rate = learning_rate)
    else:
        opt = optimizers.SGD(learning_rate = learning_rate)
    
    model.compile(optimizer=opt, loss=cost_function, metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=num_epochs, batch_size=minibatch_size)
    
def train(model, dataset, minibatch_size, num_epochs, learning_rate, beta1, beta2, epsilon, model_path, print_cost=False):
    steps_per_epoch = dataset_steps(dataset, minibatch_size)
    if optimizer == "SGD":
        opt = optimizers.SGD(learning_rate = learning_rate)
    elif optimizer == "adam":
        opt = optimizers.Adam(learning_rate = learning_rate)
    else:
        opt = optimizers.SGD(learning_rate = learning_rate)
        
    model.compile(optimizer=opt, loss=cost_function, metrics=['accuracy',Precision(), Recall()])
    model.fit_generator(data_generator(dataset, minibatch_size), steps_per_epoch=steps_per_epoch, epochs=num_epochs)
    
def predict(model, test_dataset_path):
    p = Path(test_dataset_path)
    file_sz = os.stat(test_dataset_path).st_size
    f = p.open('rb')

    if(f.tell()>= file_sz):
        f.seek(0)
    sector = np.load(f, allow_pickle=True)
    sector_X = sector[()]['X'].T # (numOfSamples, numOfFeatures)
    sector_Y = sector[()]['y'].T        
    preds = model.evaluate(x=sector_X, y=sector_Y)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    

model = MWD_lr(cf.NUM_OF_FEATURES, cost_function, optimizer, initializer, L2_lambd)
#batch = ds.load_dataset(train_dataset_path, 0)
#X_train = batch['X'].T
#X_train = X_train.reshape((X_train.shape[0], cf.WIDTH, cf.HEIGHT, 1))
#Y_train = batch['y'].T
#train(model, X_train, Y_train, MINIBATCH_SIZE, num_epochs, learning_rate, beta1, beta2, epsilon, model_path, print_cost)
train(model, train_dataset_path, MINIBATCH_SIZE, num_epochs, learning_rate, beta1, beta2, epsilon, model_path, print_cost)

# test predict
predict(model, test_dataset_path)
model.summary()
#tf.keras.utils.plot_model(model, to_file="tf2_MWD_lr", show_shapes=True)

# serialize model to JSON
model_json = model.to_json()
with open("tf2_MWD_lr.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("tf2_MWD_lr.h5")
print("Saved model to disk")


    
    
