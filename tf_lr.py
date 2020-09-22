import os
import sys
from os.path import isfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import config as cf
import dataset as ds
import Utils
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
tf.disable_v2_behavior()

# Model
cost_function="cross_entropy"
optimizer="GradientDescent"
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
lambd = 0.0
beta1 = 0.0
beta2 = 0.0
epsilon = 0.0
print_cost = True

def model(size, cost_function="cross_entropy", optimizer="GradientDescent", initializer="random"):
    '''
    default value of number of labels is 1
    '''
    model = {}
    model["size"] = size
    model["cost_function"] = cost_function
    model["optimizer"] = optimizer
    model["initializer"] = initializer
    parameters = parameters_initialize(size, initializer)
    model["parameters"] = parameters
    return model

def parameters_initialize(size, initializer = "zero"):
    tf.reset_default_graph()
    tf.set_random_seed(1)
    parameters = None
    b = tf.get_variable("b", [1, 1], initializer=tf.zeros_initializer)    
    if initializer == "zero":
        W = tf.get_variable("W", [1, size], initializer=tf.zeros_initializer)    
    elif initializer == "random":
        W = tf.get_variable("W", [1, size], initializer=tf.random_normal_initializer(seed=1))
    elif initializer == "xavier":
        W = tf.get_variable("W", [1, size], initializer=tf.xavier_initializer(seed=1))
    elif initializer == "he":
        W = tf.get_variable("W", [1, size], initializer=tf.he_normal(seed=1))
    with tf.Session() as ss:
        ss.run(tf.global_variables_initializer())
        parameters = {"w": ss.run(W),
                      "b": ss.run(b)}
        ss.close()
    return parameters

def propagate(parameters, X):
    W = parameters['w']
    b = parameters['b']
    Z = tf.add(tf.matmul(W,X), b)
    return Z

def compute_cost(Z, Y, cost_function="cross_entropy"):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = None
    if cost_function == "cross_entropy":
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

def train(model, dataset, minibatch_size, num_epochs, lambd, learning_rate, beta1, beta2, epsilon, model_path, print_cost=False):
    tf.reset_default_graph()
    
    size = model["size"]
    cost_function = model["cost_function"]
    optimizer = model["optimizer"]
    initializer = model["initializer"]
    params = model["parameters"]
    W = tf.get_variable('W', initializer=params['w'])
    b = tf.get_variable('b', initializer=params['b'])
    parameters = {'w': W,
                  'b': b}
    
    costs = []
    X = tf.placeholder(name="X", shape=(W.shape[1], None), dtype=tf.float32)
    Y = tf.placeholder(name="Y", shape=(1, None), dtype=tf.float32)
    
    # forward
    Z = propagate(parameters, X)
    cost = compute_cost(Z, Y, cost_function)
    # backward
    train = None
    if optimizer == "GradientDescent":
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    elif optimizer == "Momentum":
        train = tf.train.MomentumOptimizer(learning_rate=learning_rate).minimize(cost)
    elif optimizer == "RMSProp":
        train = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    elif optimizer == "Adam":
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    p = Path(dataset)
    file_sz = os.stat(dataset).st_size
    f = p.open('rb')
    
    with tf.Session() as ss:
        ss.run(init)
        # Do the training loop of epochs
        for epoch in range(num_epochs):
            f.seek(0)
            epoch_cost = 0.
            while f.tell() < file_sz:
                batch = np.load(f, allow_pickle=True)
                batch_X = batch[()]['X']
                batch_Y = batch[()]['y']
                numOfMiniBatch = batch_X.shape[1]//minibatch_size
                if numOfMiniBatch*minibatch_size < X.shape[1]:
                    numOfMiniBatch = numOfMiniBatch + 1        
                for m in range(0,numOfMiniBatch):
                    start = m*minibatch_size
                    end = min((m+1)*minibatch_size, X.shape[1])
                    minibatch_X = batch_X[:,start:end]
                    minibatch_Y = batch_Y[:,start:end]
                    _, minibatch_cost = ss.run([train, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                    epoch_cost += minibatch_cost / minibatch_size
            # Print the cost every epoch
            if epoch % 10 == 0:
                if print_cost:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                parameters['w'] = ss.run(W)
                parameters['b'] = ss.run(b)
                model["parameters"] = parameters
                model['minibatch_size'] = minibatch_size
                model['num_epochs'] = num_epochs
                model['lambda'] = lambd
                model['learning_rate'] = learning_rate
                model['beta1'] = beta1
                model['beta2'] = beta2
                model['epsilon'] = epsilon
                with open(model_path, 'wb') as out_file:
                    np.save(out_file, model)
                    out_file.close()
                    print("Model saved!")
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)
        # end of loop of epochs
        parameters['w'] = ss.run(W)
        parameters['b'] = ss.run(b)
        ss.close()

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    f.close()
    return parameters, costs

model = model(cf.NUM_OF_FEATURES, cost_function, optimizer, initializer)
params, costs = train(model, train_dataset_path, MINIBATCH_SIZE, num_epochs, lambd, learning_rate, beta1, beta2, epsilon, model_path, print_cost)
