import os
import sys
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

# Traning
MINIBATCH_SIZE = 10000
learning_rate = 0.1
iteration = 1000
beta1 = 0.0
beta2 = 0.0
epsilon = 0.0
print_cost = True

model = lr.model(cf.NUM_OF_FEATURES, cost_function, optimizer, initializer)
batch = ds.load_dataset(cf.out_mixed_dataset, 0) # load dataset batch position 0
params, costs = lr.train(model, batch['X'], batch['y'], iteration, learning_rate, beta1, beta2, epsilon, "./MODEL/bbb_lr_1.npy", print_cost)

#batch = ds.load_dataset(cf.out_mixed_dataset, 1) # load dataset batch position 0
#model = lr.load_model("./MODEL/bbb_lr_1.npy")
#params, costs = lr.retrain(model, batch['X'], batch['y'], iteration, learning_rate, beta1, beta2, epsilon, "./MODEL/bbb_lr_1.npy", print_cost)
