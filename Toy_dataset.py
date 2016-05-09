'''
    Written by Amr Koura.
    This script will include code that solve questions 2 in assignment 2.
'''
import numpy as np
import random
import matplotlib.pyplot as plt

def build_data_clusters():
    '''
        This function will build clusters of data points arround the following 
        Centers:
        means=[(-2,2),(-1,2),(-1,1),(-2,1),(1,2),(2,2),(2,1),(1,1),
        (-2,-1),(-1,-1),(-1,-2),(-2,-2),(1,-1),(2,-1),(2,-2),(1,-2)]
        each cluster will have 400 points that are drawn according to
        Gaussian distribution with the corrsongind means and the following :
        respective covariance matrix from the fol-lowing list:
        cov = [diag([0.1,0.1]), diag([0.15,0.07]), diag([0.15,0.07]), diag([0.1,0.1]),diag([0.1,0.1]),
        diag([0.15,0.07]), diag([0.15,0.07]), diag([0.1,0.1]),diag([0.1,0.1]), diag([0.15,0.07]),
        diag([0.15,0.07]), diag([0.1,0.1]),diag([0.1,0.1]), diag([0.15,0.07]), diag([0.15,0.07]),
        diag([0.1,0.1])]
    return:
        list of 16 arrays , each array contains 400 elements with dimintion 2 , and
        labels
    '''
    print 'start'
    means=[(-2,2),(-1,2),(-1,1),(-2,1),(1,2),(2,2),(2,1),(1,1),
        (-2,-1),(-1,-1),(-1,-2),(-2,-2),(1,-1),(2,-1),(2,-2),(1,-2)]
    
    cov = [np.diag([0.1,0.1]), np.diag([0.15,0.07]), np.diag([0.15,0.07]), np.diag([0.1,0.1]),np.diag([0.1,0.1]),
        np.diag([0.15,0.07]), np.diag([0.15,0.07]), np.diag([0.1,0.1]),np.diag([0.1,0.1]), np.diag([0.15,0.07]),
        np.diag([0.15,0.07]), np.diag([0.1,0.1]),np.diag([0.1,0.1]), np.diag([0.15,0.07]), np.diag([0.15,0.07]),
        np.diag([0.1,0.1])]
    
    clusters=[]
    labels=[]
    for i in range(len(means)):
        clusters.append(np.random.multivariate_normal(means[i],cov[i],400))
        labels.append(i%2)
    return clusters,labels

def main():
    clusters,labels=build_data_clusters()
    #visualize the data
    for i in range(len(clusters)):
        if i %2==0:
            if i==0: # want to make sure that the labels of class will be written only once
                plt.plot(clusters[i][:,0], clusters[i][:,1], 'ro',  marker='*', label='class 1')
            else:
                plt.plot(clusters[i][:,0], clusters[i][:,1], 'ro',  marker='*')
    
        else:
            if i==1: # want to make sure that the labels of class will be written only once
                plt.plot(clusters[i][:,0], clusters[i][:,1], 'bo',  marker='*', label='class 2')
            else:
                plt.plot(clusters[i][:,0], clusters[i][:,1], 'bo',  marker='*')        
    # save the figure
    plt.title('Toy example')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='upper left')
    plt.savefig('Toy Example.png')

if __name__ == '__main__':
    main() 