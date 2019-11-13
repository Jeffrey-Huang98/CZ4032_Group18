import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import preprocessing

# preprocess data
trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False)

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

ax1.scatter(X[:,0],X[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('k-means clustering plot')

ax2.scatter(X[:,0],X[:,1],  c = category, cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')

plt.show()