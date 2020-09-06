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
optimizer="GradientDecent"
initializer="random"
model_output_1="./MODEL/bbb_lr_1.npy"
model_output_2="./MODEL/bbb_lr_2.npy"

# Parameter
MINIBATCH_SIZE = 10000
learning_rate = 0.2
iteration = 10000
beta1 = 0.0
beta2 = 0.0
epsilon = 0.0
print_cost = True

# Training
# Mixed dataset
'''
model = lr.model(cf.NUM_OF_FEATURES, cost_function, optimizer, initializer)
batch = ds.load_dataset(cf.out_mixed_dataset, 0) # load dataset batch position 0
params, costs = lr.train(model, batch['X'], batch['y'], iteration, learning_rate, beta1, beta2, epsilon, model_output_1, print_cost)
'''

'''
batch = ds.load_dataset(cf.out_mixed_dataset, 0)
model = lr.load_model(model_output_1)
params, costs = lr.train(model, batch['X'], batch['y'], iteration, learning_rate, beta1, beta2, epsilon, model_output_1, print_cost)
'''

#masked dataset
'''
batch = ds.load_dataset(cf.out_masked_dataset, 0)
model = lr.load_model(model_output_1)
params, costs = lr.train(model, batch['X'], batch['y'], iteration, learning_rate, beta1, beta2, epsilon, model_output_1, print_cost)
'''

# Predict
#batch = ds.load_dataset(cf.out_mixed_dataset, 0)
batch = ds.load_dataset(cf.out_masked_dataset, 0)
model = lr.load_model(model_output_1)
params = {}
params['w'] = model['w']
params['b'] = model['b']
Y_pred = lr.predict(params, batch['X'])
Y =  batch['y']
paths = batch['paths']
np.set_printoptions(threshold=sys.maxsize)

a = paths[((Y==1)&(Y_pred==1))[0]]
b = paths[((Y==1)&(Y_pred==0))[0]]
c = paths[((Y==0)&(Y_pred==0))[0]]
d = paths[((Y==0)&(Y_pred==1))[0]]

accuracy = (len(a) + len(c))/Y.shape[1]
precision = len(a)/(len(a)+len(d))
recall = len(a)/(len(a)+len(b))

print('accuracy = ' + str(accuracy))
print('precision = ' + str(precision))
print('recall = ' + str(recall))
