 #!/usr/local/bin/python

#########################################################################
# K-Mean clustering on all training examples                            #
# Author: Ashish Arya                                                   #
# Date Created: 14/10/2017                                              #
# Purpose: 1) To perform K-means clustering on aggregate data           #
#                                                                       #
#                                                                       #
#########################################################################

import numpy as np
import pickle
import os
import warnings

path = "./data/"
print("Computing clusters...")
train = np.loadtxt(path + 'k_mean_train_melfilter48.dat')
#print(train.shape)

################   Deleting existing files   ####################
try:
	os.remove('./data/k_mean_clusters.dat')
except OSError:
	#print("File does not exist!!!")
	pass

#####################     K-Means clustering     #################

def kMeans(X, K, maxIters = 100):
    #print(len(X))
    #print(np.random.choice(np.arange(len(X)), K))
    #centroids = X[np.random.choice(np.arange(len(X)), K), :]
    centroids = np.copy(X[:K,:])
    #print(centroids)
    for i in range(maxIters):
        # Cluster Assignment step
        #with warnings.catch_warnings():
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        #print(C)
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore", category=RuntimeWarning)
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        #print(centroids)
        #print("Wait")
    #print(C)
    np.savetxt('./data/temp.dat',C,delimiter = ' ')
    return np.array(centroids)

no_clusters = 64          # Number of clusters i.e. K
centroids = kMeans(train,no_clusters)
#print(centroids[:,:5])
np.savetxt('./data/k_mean_clusters.dat',centroids)