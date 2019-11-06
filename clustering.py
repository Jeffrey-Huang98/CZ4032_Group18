import preprocessing
from copy import deepcopy
import numpy as np
import pylab as plt
import pandas as pd

# preprocess data
trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False)
print(trainX)
print(trainY)
trainX = pd.concat([trainX, trainY], axis=1)
print(trainX)

# Change categorical data to number 0-2
trainX["diagnosis"] = pd.Categorical(trainX["diagnosis"])
trainX["diagnosis"] = trainX["diagnosis"].cat.codes
# Change dataframe to numpy matrix
data = trainX.values[:, 0:30]
category = trainX.values[:, 30]

print(data)
print(category)

# Number of clusters
k = 2
# Number of training data
n = data.shape[0]
# Number of features in the data
c = data.shape[1]

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
centers = np.random.randn(k,c)*std + mean

colors=['orange', 'blue']
for i in range(n):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
plt.show()

centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
centers_new    

# Plot the data and the centers generated as random
colors=['orange', 'blue']
for i in range(n):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)
plt.show()


# trainY = pd.Categorical(trainY)
# trainY = trainY.codes
# print(len(trainY))