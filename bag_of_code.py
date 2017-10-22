 #!/usr/local/bin/python

#########################################################################
# Bag of code generation for each pair of files                         #
# Author: Ashish Arya                                                   #
# Date Created: 14/10/2017                                              #
# Purpose: 1) To generate bag of code from computed clusters            #
#                                                                       #
#                                                                       #
#########################################################################

import numpy as np
import pickle
import os
import sys

################    All Constants and paths used    #####################
path = "./data/"
n_classes = 10 # Number of classes in bird data
relativePathForTrain = "./data/melfilter48/train/"
testFilesExtension = '.mfcc'
clusterFile = './data/k_mean_clusters.dat'

################    Loading cluster file    #######################
centroids = np.loadtxt(clusterFile)
n_clusters = centroids.shape[0]
n_input = centroids.shape[1] # input dimensionality ----> 585
#print(n_clusters)
#print(n_input)

################   Deleting existing files   ####################
try:
	os.remove('./data/train_melfilter48.dat')
except OSError:
	#print("File does not exist!!!")
	pass

try:
	os.remove('./data/train_mel48.dat')
except OSError:
	#print("File does not exist!!!")
	pass

################    Creating pair wise frames    #######################

print("Computing....")
print("Generating pair wise bag of code ....")
f_handle = open('./data/train_mel48.dat', 'ab')
if len(sys.argv) == 1:
	for root, dirs, files in os.walk(relativePathForTrain, topdown=False):
		###print(dirs)
		#for name in dirs:
		#print("Class Name : ",name)
		for k in range(n_classes):
			for j in range(n_classes):
				if j > k: 
					for name in dirs:
						if name == dirs[k]:
							###print("Class Name : ",dirs[k])
							parts = []
							parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]
							#print(parts)
							for part in parts:
								###print('')
								###print("Part : ",part)
								example = np.loadtxt(os.path.join(root,name,part))
								i = 0
								rows, cols = example.shape
								#print(rows)
								#print(cols)
								context = np.zeros((rows-14,15*cols)) # 15 contextual frames
								while i <= (rows - 15):
									ex = example[i:i+15,:].ravel()
									ex = np.reshape(ex,(1,ex.shape[0]))
									context[i:i+1,:] = ex
									i += 1
								r, c = context.shape 			# rows and columns of context 
								#print("Shape : ",r , " ",c)
								#print(context)
								bagOfcode = np.zeros(n_clusters)
								# Calculating the bag of code
								C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in context])
								#print(C)
								for i in range(r):
									bagOfcode[C[i]] += 1
								bag1 = bagOfcode
								###print(bag1)
								actual_class_temp1 = name.split('c')
								#print(actual_class_temp)
								actual_class_temp1.pop(0)
								actual_class1 = []
								for i in actual_class_temp1:
									actual_class1.append(int(i))
								#print(actual_class1)

								for name2 in dirs:
									if name2 == dirs[j]:
										###print("Class Name : ",dirs[j])
										#print(name,"--",name2)
										parts2 = []
										parts2 += [each for each in os.listdir(os.path.join(root,name2)) if each.endswith(testFilesExtension)]
										###print(parts2)
										for part2 in parts2:
											#print("Part : ",part2)
											example = np.loadtxt(os.path.join(root,name2,part2))
											i = 0
											rows, cols = example.shape
											#print(rows)
											#print(cols)
											context = np.zeros((rows-14,15*cols)) # 15 contextual frames
											while i <= (rows - 15):
												ex = example[i:i+15,:].ravel()
												ex = np.reshape(ex,(1,ex.shape[0]))
												context[i:i+1,:] = ex
												i += 1
											r, c = context.shape 			# rows and columns of context 
											###print("Shape : ",r , " ",c)
											#print(context)
											bagOfcode = np.zeros(n_clusters)
											# Calculating the bag of code
											C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in context])
											#print(C)
											for i in range(r):
												bagOfcode[C[i]] += 1
											bag2 = bagOfcode
											###print(bag2)
											finalBag = np.zeros(n_clusters)
											for m in range(n_clusters):
												finalBag[m] = bag1[m]+bag2[m]
											###print("B1-B2 : ",finalBag)
											actual_class_temp2 = name2.split('c')
											#print(actual_class_temp)
											actual_class_temp2.pop(0)
											actual_class2 = []
											for i in actual_class_temp2:
												actual_class2.append(int(i))
											#print(actual_class2)
											###print(actual_class1[0], ' ', actual_class2[0])
											finalBag = np.append(finalBag,actual_class1[0])
											finalBag = np.append(finalBag,actual_class2[0])
											###print(finalBag)
											finalBag = np.reshape(finalBag,(1,finalBag.shape[0]))
											np.savetxt(f_handle, finalBag)
							###print('')
					###print('')
					###print('')
					###print('')
f_handle.close()

A = np.loadtxt('./data/train_mel48.dat')
np.random.shuffle(A)
np.savetxt('./data/train_melfilter48.dat',A)




## for generating bag of code for only single file
'''if len(sys.argv) == 1:
	for root, dirs, files in os.walk(relativePathForTrain, topdown=False):
		print(dirs)
		for name in dirs:
			print("Class Name : ",name)
			parts = []
			parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]
			print(parts)
			for part in parts:
				print("Part : ",part)
			example = np.loadtxt(os.path.join(root,name,part))
			i = 0
			rows, cols = example.shape
			#print(rows)
			#print(cols)
			context = np.zeros((rows-14,15*cols)) # 15 contextual frames
			while i <= (rows - 15):
				ex = example[i:i+15,:].ravel()
				ex = np.reshape(ex,(1,ex.shape[0]))
				context[i:i+1,:] = ex
			i += 1
			r, c = context.shape 			# rows and columns of context 
			print("Shape : ",r , " ",c)
			#print(context)
			bagOfcode = np.zeros(n_clusters)
			# Calculating the bag of code
			C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in context])
			#print(C)
			for i in range(r):
				bagOfcode[C[i]] += 1
			bag1 = bagOfcode
			print(bag1)'''


					