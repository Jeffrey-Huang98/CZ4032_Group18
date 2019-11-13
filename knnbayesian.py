import preprocessing
import pylab as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# preprocess data
trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False)

# k-Nearest Neighbour
K = [3, 5, 7, 11, 13, 17, 19, 23]
knn_accuracy = []

for k in K:
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(trainX, trainY)
    predicted = model.predict(testX)
    acc = accuracy_score(testY, predicted)
    knn_accuracy.append(acc)

plt.figure()
plt.plot(range(len(K)), knn_accuracy)
plt.xticks(range(len(K)), K)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title("Accuracy against k")
plt.savefig('kNN.png')
plt.show()

print("kNN accuracy:", knn_accuracy)


# Naive Bayesian
model = GaussianNB()
model.fit(trainX, trainY)
predicted = model.predict(testX)

print("")
print("Naive Bayesian:")
print(predicted)

print("Accuracy Naive Bayesian:", accuracy_score(testY, predicted))