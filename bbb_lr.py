import os
import sys
import numpy as np
import config as cf
import dataset as ds
import Utils

# BBBLearning lib
BBBLearning_PATH = '../BBBLearning'
sys.path.append(BBBLearning_PATH)

from BBBLearning.lr import lr

# Model
cost_function="cross_entropy"
optimizer="GradientDescent"
initializer="zero"
model_output_1="./MODEL/bbb_lr_1.npy"
model_output_2="./MODEL/bbb_lr_2.npy"
balanced_model_output_1="./MODEL/bbb_lr_balanced_1.npy"
balanced_model_output_2="./MODEL/bbb_lr_balanced_2.npy"
balanced_model_output_3="./MODEL/bbb_lr_balanced_3.npy"
zero_init_model_output="./MODEL/bbb_lr_zero_init.npy"
model_path = zero_init_model_output

# Dataset
#train_dataset_path = cf.out_masked_dataset
#train_dataset_path = cf.out_face_dataset
#train_dataset_path = cf.out_mixed_dataset
#train_dataset_path = cf.out_mixed_balanced_dataset
train_dataset_path = cf.out_mixed_balanced_dataset_train

#test_dataset_path = cf.out_mixed_dataset
test_dataset_path = cf.out_mixed_balanced_dataset_test

# Parameters
MINIBATCH_SIZE = 100
learning_rate = 0.1
num_epochs = 10000
lambd = 0.0
beta1 = 0.0
beta2 = 0.0
epsilon = 0.0
print_cost = True

# Training
model = lr.model(cf.NUM_OF_FEATURES, cost_function, optimizer, initializer)
#model = lr.load_model(model_path)
batch = ds.load_dataset(train_dataset_path, 0)
params, costs = lr.train(model, batch['X'], batch['y'], MINIBATCH_SIZE, num_epochs, lambd, learning_rate, beta1, beta2, epsilon, model_path, print_cost)
#params, costs = lr.train2(model, train_dataset_path, MINIBATCH_SIZE, num_epochs, lambd, learning_rate, beta1, beta2, epsilon, model_path, print_cost)


# Predict
batch = ds.load_dataset(test_dataset_path, 0)
model = lr.load_model(model_path)
params = model['parameters']
Y_pred = lr.predict(params, batch['X'])
Y =  batch['y']
paths = batch['paths']
Utils.skewed_error_analysis(Y, Y_pred, paths, printpath=False)
