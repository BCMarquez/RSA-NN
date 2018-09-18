import argparse
import sys

import operator 
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle
import tensorflow as tf
import math
import scipy.spatial.distance
import csv
import softmax
import time
from sympy.utilities.iterables import multiset_permutations
from datetime import datetime

np.set_printoptions(formatter={'float_kind':'{:.3f}'.format})


n_dims = 10

train_data = "data//train.json"
dev_data = "data//dev.json"
test_data = "data//test.json"


#Beta for regularization
beta = .0000001


def pretty_print(env_data, path):
    with open(path,'a') as pretty_print:
        permed_objs = multiset_permutations(env_data["env"])
        for env,label,pred in zip(permed_objs,env_data["y"],env_data["y_hat"]):
            utter = env_data["utter"]
            obj1 = env[0]["color"] + env[0]["shape"]
            obj2 = env[1]["color"] + env[1]["shape"]
            obj3 = env[2]["color"] + env[2]["shape"]
            pretty_print.write("\"%s\": %s| %s| %s -> %s -> %s\n"
                %(utter,obj1, obj2, obj3, label, pred))


def conv_to_csv(matrix):
    returns_path = "viz.csv"
    headers = []
    emb_weights = []
    file = open(returns_path,'w')
    writer = csv.writer(file)
    for i in range(n_dims):
        headers.append("x"+str(i))   #create headers
    writer.writerow(headers)

    for word_embedding in matrix:
        for weight in word_embedding:
            emb_weights.append(weight)
        writer.writerow(emb_weights)
        emb_weights = []

def top_weights(matrix):
    print("weight matrix: ", matrix)
    for i in range(matrix.shape[1]):
        col = matrix[:,i]
        top4 = np.argpartition(col, -4)[-4:]
        print("top4: ",top4)

def model(oneHotVecs,weights):
    h1 = tf.matmul(oneHotVecs,weights["layer_one"])
    h1 = tf.add(h1,weights["layer_one_bias"])
    h1 = tf.nn.tanh(h1)
    h2 = tf.matmul(h1,weights["layer_two"])
    h2 = tf.add(h2,weights["layer_two_bias"])
    h2 = tf.nn.sigmoid(h2)
    h3 = tf.matmul(h2,weights["layer_three"])
    h3 = tf.add(h3,weights["layer_three_bias"])
    return h3

ex_inst = eval(open(train_data).readline())
n_features = len(ex_inst["x"][0])
n_states = len(ex_inst["y"][0])

def main():
    sess = tf.InteractiveSession()

    #Input Placeholders
    inputDataset = tf.placeholder(tf.float32, shape = [None, n_features]) #consider changing it from float to int
    outputLabels = tf.placeholder(tf.float32, shape = [None, n_states])

    #Variables
    weights = {
            "layer_one" : tf.get_variable(
                "layer_one", shape = [n_features,50],initializer = tf.contrib.layers.xavier_initializer()), 
            "layer_one_bias" : tf.Variable(tf.zeros(50)),

            "layer_two" : tf.get_variable(
                "layer_two", shape = [50,30], initializer = tf.contrib.layers.xavier_initializer()),
            "layer_two_bias" : tf.Variable(tf.zeros(30)),

            "layer_three" : tf.get_variable(
                "layer_three", shape = [30,n_states], initializer = tf.contrib.layers.xavier_initializer()),
            "layer_three_bias" : tf.Variable(tf.zeros(n_states))
    }

    nnModel = model(inputDataset,weights)

    #Loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = outputLabels,logits = nnModel)
    regularizer = tf.nn.l2_loss(weights["layer_one"])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = .01).minimize(loss+ regularizer * beta)
    init = tf.global_variables_initializer()
    sess.run(init)

    
    for step in range(150):
        for i,env_data in enumerate(open(train_data,'r')):
            env_data = eval(env_data)
            x_train = env_data["x"] #check if it's a numpy array
            y_train = env_data["y"] 
            loss_value,_ = sess.run([loss,optimizer],feed_dict = {inputDataset:x_train,outputLabels:y_train})
            if step % 10 == 0:
                print("Step ", step," Average Loss: ", sum(loss_value)/len(loss_value),end='\r')
                

    y_hat= tf.placeholder(tf.float32, shape = [None, n_states])

#--------- Accuracy Metric ------------------------------
    correct_prediction = tf.equal(tf.argmax(nnModel,1), tf.argmax(outputLabels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#----------- Mean Squared Error Metric -------------------------------
# USING SKLEARN FOR NOW. FUTURE ONE WILL BE IN TENSORFLOW

#----------- Cross Entropy Metric ----------------------------------
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(outputLabels * tf.log(y_hat), reduction_indices=[1])) #should I be taking the mean after I've aggregated all of them?

#-------------- Training Metrics --------------------------------
    write_train = "predictions\\hidden_env_train.txt"
    count = 0
    aggregated_acc = 0
    aggregated_MSE = 0
    aggregated_CE = 0

    startTime  = dateTime.now()
    for i,env_data in enumerate(open(train_data,'r')):
        env_data = eval(env_data)
        x_train = env_data["x"] #check if it's a numpy array
        y_train = env_data["y"] 
        predictions = sess.run([nnModel], feed_dict = {inputDataset:x_train, outputLabels:y_train})
        predictions = softmax.softmax(np.array(predictions[0]), axis = 1)
        aggregated_acc += float(accuracy.eval({inputDataset:x_train,outputLabels:y_train}))
        aggregated_MSE += float(metrics.mean_squared_error(y_train,predictions))
        aggregated_CE += float(cross_entropy.eval({outputLabels:y_train, y_hat:predictions}))
        count += 1 #there are 6 instances per environment. Should I be incrementing by 6?
        env_data["y_hat"] = predictions
        pretty_print(env_data, write_train)
        
    print("time to predict training: "% datetime.now()-startTime)

    avg_acc = aggregated_acc/count
    avg_MSE = aggregated_MSE/count  
    avg_CE = aggregated_CE/count  

    open(write_train,'a').write("avg_acc = %s\n"% avg_acc)
    open(write_train,'a').write("avg_MSE = %s\n"% avg_MSE)
    open(write_train,'a').write("avg_CE  = %s\n"% avg_CE)

    print("training avg acc: ", avg_acc)
    print("training avg mse", avg_MSE)
    print("training avg ce", avg_CE)


#-------------- Dev Metrics --------------------------------
    write_dev = "predictions\\hidden_env_dev.txt"
    count = .0000000000000000000001
    aggregated_acc = 0
    aggregated_MSE = 0
    aggregated_CE = 0

    
    startTime  = dateTime.now()
    for i,env_data in enumerate(open(dev_data,'r')):
        env_data = eval(env_data)
        x_dev = env_data["x"] #check if it's a numpy array
        y_dev = env_data["y"] 
        predictions = sess.run([nnModel], feed_dict = {inputDataset:x_dev, outputLabels:y_dev})
        predictions = softmax.softmax(np.array(predictions[0]), axis = 1)
        aggregated_acc += float(accuracy.eval({inputDataset:x_dev,outputLabels:y_dev}))
        aggregated_MSE += float(metrics.mean_squared_error(y_dev,predictions))
        aggregated_CE += cross_entropy.eval({outputLabels:y_dev, y_hat:predictions})
        count += 1
        env_data["y_hat"] = predictions
        pretty_print(env_data, write_dev)
        
    
    print("time to predict dev: "% datetime.now()-startTime)

    avg_acc = aggregated_acc/count
    avg_MSE = aggregated_MSE/count
    avg_CE = aggregated_CE/count

    open(write_dev,'a').write("avg_acc = %s\n"% avg_acc)
    open(write_dev,'a').write("avg_MSE = %s\n"% avg_MSE)
    open(write_dev,'a').write("avg_CE  = %s\n"% avg_CE)

    print("dev avg acc: ",avg_acc)
    print("dev avg mse: ", avg_MSE)
    print("dev avg ce: ", avg_CE)

#-------------- Test Metrics --------------------------------
    write_dev = "predictions\\hidden_env_test.txt"
    count = .0000000000000000000001
    aggregated_acc = 0
    aggregated_MSE = 0
    aggregated_CE = 0

    startTime  = dateTime.now()
    for i,env_data in enumerate(open(test_data,'r')):
        env_data = eval(env_data)
        x_dev = env_data["x"] #check if it's a numpy array
        y_dev = env_data["y"] 
        predictions = sess.run([nnModel], feed_dict = {inputDataset:x_dev, outputLabels:y_dev})
        predictions = softmax.softmax(np.array(predictions[0]), axis = 1)
        aggregated_acc += float(accuracy.eval({inputDataset:x_dev,outputLabels:y_dev}))
        aggregated_MSE += float(metrics.mean_squared_error(y_dev,predictions))
        aggregated_CE += cross_entropy.eval({outputLabels:y_dev, y_hat:predictions})
        count += 1
        env_data["y_hat"] = predictions
        pretty_print(env_data, write_dev)
        
    
    print("time to predict test: "% datetime.now()-startTime)

    avg_acc = aggregated_acc/count
    avg_MSE = aggregated_MSE/count
    avg_CE = aggregated_CE/count

    open(write_dev,'a').write("avg_acc = %s\n"% avg_acc)
    open(write_dev,'a').write("avg_MSE = %s\n"% avg_MSE)
    open(write_dev,'a').write("avg_CE  = %s\n"% avg_CE)

    print("test avg acc: ",avg_acc)
    print("test avg mse: ", avg_MSE)
    print("test avg ce: ", avg_CE)
    #conv_to_csv(weights["layer_one"].eval())
    #top_weights(weights["layer_one"].eval())


if __name__ == '__main__':
    main()
