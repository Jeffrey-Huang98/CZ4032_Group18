import pandas as pd
import preprocessing
from sklearn.cluster import KMeans
import pylab as plt
import numpy as np
from itertools import cycle, islice

x_train, x_test, y_train, y_test = preprocessing.preprocess(0.3, False)
x_train = x_train.values

kmeans = KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(x_train)
centers = kmeans.cluster_centers_

# plot the clusters
plt.scatter(x_train[y_kmeans == 0, 0], x_train[y_kmeans == 0, 1], s = 7, c = 'red', label = '0')
plt.scatter(x_train[y_kmeans == 1, 0], x_train[y_kmeans == 1, 1], s = 7, c = 'blue', label = '1')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Center')

plt.legend()
plt.show()