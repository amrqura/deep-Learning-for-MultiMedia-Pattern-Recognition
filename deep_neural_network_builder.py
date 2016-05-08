'''

    Written by Amr Koura
    this script will contains code that will build the deep neural network
    according to the parameters from user. the parameter will contains the
    following attributes:
    1- # of input neurons
    2- # of output neurons
    3- list of hidden layer sizes , e.g[1024,256,1024]
    4- activation function for the hidden neurons
    5- activation function for the output neurons

'''
import tensorflow as tf
import sys
import numpy as np
import math

def build_deep_neural_network(input_size,output_size,hidden_layer_sizes,hidden_layer_activation,output_activation_function):
    '''
    build the deep neural network in dynamic way. 
    Paramter:
        input_size: size of input layer
        output_size: size of output layer
        hidden_layer_sizes: list contains the size of corresponding layer, # of hidden layers= length of list
        hidden_layer_activation= activation function for the hidden layers.
        output_activation_function: activation function for the output layer
    return:
        it will return the tensor that correspond to the output layer, and this will
        will be in top of execution graph
    '''
    print 'building neural network ......'
    # start with input placeholder, size will bi equals to input_size
    X=tf.placeholder(tf.float32,shape=(None,input_size))
    # build the placeholder for the for the output layer
    Y=tf.placeholder(tf.float32,shape=(None,output_size))
    # build the hidden layers
    num_of_hidden_layers=len(hidden_layer_sizes)
    #array of Weights
    W=[]
    #biases variables = # of hidden layers +1: because the output layer will contains biases as well
    biases=[]
    #hidden layer activation functions =# of hidden layers
    hidden_functions=[]
    
    # weights should be tensorflow variables since they will be updated by the learning algorithm
    previous_layer_size=input_size
    privious_input=X
    for i in range(num_of_hidden_layers):
        W.append(tf.Variable(
            tf.truncated_normal([previous_layer_size, hidden_layer_sizes[i]],
                                stddev=1.0 / math.sqrt(float(previous_layer_size))),
            name='weights'+str(i)))
        biases.append(tf.Variable(tf.zeros([hidden_layer_sizes[i]]),
                             name='biases'+str(i)))
        previous_layer_size=hidden_layer_sizes[i]
        hidden_functions.append(hidden_layer_activation(tf.matmul(privious_input, W[i]) + biases[i]))
        # to be used in the next iterations
        privious_input=hidden_functions[i]
    
    #output layer
    W.append(tf.Variable(
            tf.truncated_normal([previous_layer_size, output_size],
                                stddev=1.0 / math.sqrt(float(previous_layer_size))),
            name='weights_output'))
    biases.append(tf.Variable(tf.zeros([output_size]),
                             name='biases_output'))
    
    logits = output_activation_function(tf.matmul(privious_input, W[num_of_hidden_layers])
                                         + biases[num_of_hidden_layers] )
    
    return logits
   
def main():
    input_size=18
    output_size=9
    hidden_layer_sizes=[36,400]
    #hidden_layer_sizes=np.array(hidden_layer_sizes.strip(']').strip('[').split(','))
    #hidden_layer_sizes=hidden_layer_sizes.astype(np.float32)
    hidden_layer_activation=tf.nn.relu
    output_activation_function=tf.nn.softmax
    logits=build_deep_neural_network(input_size, output_size, hidden_layer_sizes, hidden_layer_activation, output_activation_function)
    print logits

if __name__ == '__main__':
    main()
