'''
    created by Amr Koura.
    In this I will show the tensorflow code for the linear regression.
    the code will contain single neuron. single neuron can can work with linearly seprable dataset.
    the provided data set , will be in the following form:
    x1=(-2,1), y1=-1
    x2=(-1,-2),y2=-1
    x3=(-1,2),y3=1
    x4=(1,1),y4=1
    
    as shown the input data set has two features and single label. the task is to create single neuron
    that can be trained by 20 randomly perturbed version of the previous data set , namely create 20 data
    set similar to the previous data set wit noise.

'''

import tensorflow as tf
import numpy as np
import random
'''
create input dataset of shape (80,2) and labels of shape (80) similar to previous
dataset with small noise.
'''
def build_train_set():
    x_data=np.zeros((80,2))
    y_data=np.zeros(80)
    for step in range(20):
        print "creating samples "+str(step)
    # generate noise:  number between -0.03 and 0.03
        r=np.float32(random.uniform(-0.03, 0.03))
        x1=[-2+r,1+r]
        y1=-1
        x_data[step*4 ]=x1
        y_data[step*4]=y1
        print "x1 is "+str(x1)
            
        x2=[-1+r,-2+r]
        y2=-1
        print "x2 is "+str(x2)
        #x_data.append(x2)
        #y_data.append(y2)
        x_data[step*4 +1]=x2
        y_data[step*4 +1]=y2
    
        x3=[-1+r,2+r]
        y3=1
        #x_data.append(x3)
        #y_data.append(y3)
        x_data[step*4 +2]=x3
        y_data[step*4 +2]=y3
        print "x3 is "+str(x3)
            
        x4=[1+r,1+r]
        y4=1
        print "x4 is "+str(x4)
        #x_data.append(x4)
        #y_data.append(y4)
        x_data[step*4 +3]=x4
        y_data[step*4 +3]=y4
        
    return x_data.astype(np.float32),y_data.astype(np.float32)

'''
    build the test data set that has 4 points with the corresponding labels.
    you can specify as much examples as possible , but just follow the same 
    dataset structure.
'''
def build_test_set():
    x_data=np.zeros((5,2))
    y_data=np.zeros(5)
    x1=[-2.19,0.99]
    x_data[0]=x1
    y_data[0]=-1
    
    x2=[-1.01,-2.02]
    x_data[1]=x2
    y_data[1]=-1
    
    x3=[-0.999,2.0083]
    x_data[2]=x3
    y_data[2]=1
    
    x4=[1.001,0.998]
    x_data[3]=x4
    y_data[3]=1
    
    x5=[-2.0001,1.000002]
    x_data[4]=x5
    y_data[4]=-1
    
    return x_data.astype(np.float32),y_data.astype(np.float32)



def main():
    # start with creating the variables
    W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    #placeholders
    X=tf.placeholder(tf.float32,shape=(None,2))
    
    #place holder for labels
    y_=tf.placeholder(tf.float32,shape=(None,1))
    #prediction operation
    y=tf.nn.sigmoid(tf.matmul(X,W)+b)
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
            print" in setp="+str(step)+" the W="+ str(sess.run(W))+" and b="+ str(sess.run(b))
    
    # testing
    test_data,test_labels=build_test_set()
    '''
    compute y will for the test data will show much differnce between the + and - values.
    for example for the above test dataset , I get the following values
    array([[  5.72775316e-04],
       [  3.43822279e-07],
       [  9.62653756e-01],
       [  9.99880791e-01],
       [  1.59412483e-03]], dtype=float32)
       
       this can show that the (+) values will get bigger values > 6 for example , while the nigative get smaller values
    
    '''
    print (sess.run(y,feed_dict={X:test_data}))
    
    
if __name__ == '__main__':
    main()    
        
