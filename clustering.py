# import preprocessing
# from copy import deepcopy
# import numpy as np
# import pylab as plt
# import pandas as pd

# # preprocess data
# trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False)
# # print(trainX)
# # print(trainY)
# # trainX = pd.concat([trainX, trainY], axis=1)
# # print(trainX)

# # # Change categorical data to number 0-2
# # trainX["diagnosis"] = pd.Categorical(trainX["diagnosis"])
# # trainX["diagnosis"] = trainX["diagnosis"].cat.codes

# trainY = pd.Categorical(trainY).codes

# # Change dataframe to numpy matrix

# # data = trainX.values[:, 0:30]
# # category = trainX.values[:, 30]

# data = trainX.values
# category = trainY


# print(data)
# print(category)

# # Number of clusters
# k = 2
# # Number of training data
# n = data.shape[0]
# # Number of features in the data
# c = data.shape[1]

# # Generate random centers, here we use sigma and mean to ensure it represent the whole data
# mean = np.mean(data, axis = 0)
# std = np.std(data, axis = 0)
# centers = np.random.randn(k,c)*std + mean

# colors=['red', 'blue']
# for i in range(n):
#     plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])
# plt.scatter(centers[:,0], centers[:,1], marker='*', c='yellow', s=150)
# plt.show()

# centers_old = np.zeros(centers.shape) # to store old centers
# centers_new = deepcopy(centers) # Store new centers

# # data.shape
# clusters = np.zeros(n)
# distances = np.zeros((n,k))

# error = np.linalg.norm(centers_new - centers_old)

# # When, after an update, the estimate of that center stays the same, exit loop
# while error != 0:
#     # Measure the distance to every center
#     for i in range(k):
#         distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
#     # Assign all training data to closest center
#     clusters = np.argmin(distances, axis = 1)
    
#     centers_old = deepcopy(centers_new)
#     # Calculate mean for every cluster and update the center
#     for i in range(k):
#         centers_new[i] = np.mean(data[clusters == i], axis=0)
#     error = np.linalg.norm(centers_new - centers_old)
# # centers_new    

# # Plot the data and the centers generated as random
# colors=['red', 'blue']
# for i in range(n):
#     plt.scatter(data[i, 0], data[i,1], s=10, color = colors[int(category[i])])
# plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='yellow', s=150)
# plt.show()


# # trainY = pd.Categorical(trainY)
# # trainY = trainY.codes
# # print(len(trainY))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.cm as cm
# %matplotlib inline
import preprocessing
from subprocess import check_output


# preprocess data
trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False)

# X = trainX.values
# np.append(X, testX.values)
# # X = np.append(trainX.values, testX.values)
# category = trainY
# np.append(category, testY)

X = pd.concat([trainX, testX], ignore_index=True)
X = X.values
category = pd.concat([trainY, testY], ignore_index=True)

#Creating a 2D visualization to visualize the clusters
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1, perplexity=40, n_iter= 4000)
Y = tsne.fit_transform(X)

#Cluster using k-means
from sklearn.cluster import KMeans
kmns = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
kY = kmns.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('k-means clustering plot')

ax2.scatter(Y[:,0],Y[:,1],  c = category, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')

plt.show()