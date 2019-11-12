import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# preprocess data
trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False)

model = KNeighborsClassifier(n_neighbors=11, weights='distance')
model.fit(trainX, trainY)
predicted = model.predict(testX)

print("KNN:")
print(predicted)

print("Accuracy KNN:", accuracy_score(testY, predicted))


model = GaussianNB()
model.fit(trainX, trainY)
predicted = model.predict(testX)

print("")
print("Naive Bayesian:")
print(predicted)

print("Accuracy Naive Bayesian:", accuracy_score(testY, predicted))