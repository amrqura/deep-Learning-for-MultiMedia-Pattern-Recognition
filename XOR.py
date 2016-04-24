'''
    created by Amr Koura.
    In this script , I will create simple neural network (with one layer) to 
    train the neural network for XOR operation.
'''

import tensorflow as tf
import numpy as np
import random

'''
create input dataset of shape (80,2) and labels of shape (80) similar to XOR
dataset with small noise.
'''
def build_train_set():
    x_data=np.zeros((80,2))
    y_data=np.zeros(80)
    for step in range(20):
        print "creating samples "+str(step)
    # generate noise:  number between -0.03 and 0.03
        r=np.float32(random.uniform(-0.03, 0.03))
        x1=[0+r,0+r]
        y1=-1
        x_data[step*4 ]=x1
        y_data[step*4]=y1
        print "x1 is "+str(x1)
            
        x2=[0+r,1+r]
        y2=1
        print "x2 is "+str(x2)
        #x_data.append(x2)
        #y_data.append(y2)
        x_data[step*4 +1]=x2
        y_data[step*4 +1]=y2
    
        x3=[1+r,0+r]
        y3=1
        #x_data.append(x3)
        #y_data.append(y3)
        x_data[step*4 +2]=x3
        y_data[step*4 +2]=y3
        print "x3 is "+str(x3)
            
        x4=[1+r,1+r]
        y4=-1
        print "x4 is "+str(x4)
        #x_data.append(x4)
        #y_data.append(y4)
        x_data[step*4 +3]=x4
        y_data[step*4 +3]=y4
        
    return x_data.astype(np.float32),y_data.astype(np.float32)

def build_test_set():
    x_data=np.zeros((6,2))
    y_data=np.zeros(6)
    x1=[.019,.09] #(0,0)
    x_data[0]=x1
    y_data[0]=-1
    
    x2=[.01,1.02]#(0,1)
    x_data[1]=x2
    y_data[1]=1
    
    x3=[.999,.0083] #(1,0)
    x_data[2]=x3
    y_data[2]=1
    
    x4=[1.001,.998]#(1,1)
    x_data[3]=x4
    y_data[3]=-1
    
    x5=[0.0001,0.02]#(0,0)
    x_data[4]=x5
    y_data[4]=-1
    
    x6=[.0001,1.02]#(0,1)
    x_data[5]=x6
    y_data[5]=1
    
    return x_data.astype(np.float32),y_data.astype(np.float32)

'''
    define the neural network that will be trained to learn the XOR operation
    input layer : 2 neurons
    hidden layer: 3 neurons
    output layer: 1 neurons

'''
def main():
    # define the placeholder, X for input , y_ for labels
    X=tf.placeholder(tf.float32,shape=(None,2))
    y_=tf.placeholder(tf.float32,shape=(None,1))
    
    # define variables
    # W1: define weights from the input layer to the hidden layers, (2,3), b=(3)
    W1 =tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))
    b1 = tf.Variable(tf.zeros([3]))
    hidden=tf.nn.relu(tf.matmul(X,W1)+b1)
    
    # from hidden layer to output layer
    #W2: (3,2) , b2=(3)
    W2 =tf.Variable(tf.random_uniform([3, 1], -1.0, 1.0))
    b2 = tf.Variable(tf.zeros([1]))
    #y=tf.nn.sigmoid(tf.matmul(hidden,W2)+b2)
    # we want the output to be -1 , 1 , so it better not to use sigmoid function , sigmoid function
    # is used only for the positive values , where the output should be between 0,1
    y=tf.matmul(hidden,W2)+b2
    # read the data set
    x_data,y_data=build_train_set()
    # create session and initialize the variables
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    #reshape y_data
    y_data=np.reshape(y_data,(80,1))
    # compute the loss function and define the train operation
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    
    for step in xrange(2001):
        sess.run(train,feed_dict={X:x_data,y_:y_data})
        if step % 200 == 0:
            print " in setp="+str(step)+" the W1="+ str(sess.run(W1))+ " and b1=" + str(sess.run(b1))+" the W2="+ str(sess.run(W2)) + " and b2=" + str(sess.run(b2))
            #print " and b1="+ str(sess.run(b1)+" the W2="+ str(sess.run(W2))+" and b2="+ str(sess.run(b2))
    
    
    
    # start testing phase
    test_data,test_labels=build_test_set()
    # get the prediction from the test data set
    print "prediction ......."
    # minus should predict as -1 and positive should be predicted as 1
    print (sess.run(y,feed_dict={X:test_data}))
    #evaluation
    '''
    print "evaluation: ..."
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_labels=np.reshape(test_labels,(5,1))
    print(sess.run(accuracy, feed_dict={X: test_data, y_: test_labels}))
    '''
    
    
if __name__ == '__main__':
    main()      