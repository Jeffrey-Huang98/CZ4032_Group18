import preprocessing
import pylab as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


print("k-Nearest Neighbours...")
K = [3, 5, 7, 11, 13, 17, 19]
features = [10, 15, 20, 25, 30]
knn_acc = []

for k in K:
    knn_acc_ = []
    for feat in features:
        # preprocess data
        trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False, feat)

        # k-Nearest Neighbour
        model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        model.fit(trainX, trainY)
        predicted = model.predict(testX)
        acc = accuracy_score(testY, predicted)
        print('k %d, no of features %d, accuracy %g'%(k, feat, acc))
        knn_acc_.append(acc)
    print("")
    knn_acc.append(knn_acc_)


plt.figure()
plt.plot(range(len(features)), knn_acc[0], label = 'k=3')
plt.plot(range(len(features)), knn_acc[1], label = 'k=5')
plt.plot(range(len(features)), knn_acc[2], label = 'k=7')
plt.plot(range(len(features)), knn_acc[3], label = 'k=11')
plt.plot(range(len(features)), knn_acc[4], label = 'k=13')
plt.plot(range(len(features)), knn_acc[5], label = 'k=17')
plt.plot(range(len(features)), knn_acc[6], label = 'k=19')
plt.xticks(range(len(features)), features)
plt.xlabel('no of features')
plt.ylabel('accuracy')
plt.title("kNN Accuracy against no of features")
plt.legend(loc='lower right')
plt.savefig('kNN.png')


max_knn_acc = 0
best_k = 0
best_knn_feat = 0

for i in range(len(K)):
    for j in range(len(features)):
        if knn_acc[i][j] > max_knn_acc:
            max_knn_acc = knn_acc[i][j]
            best_k = K[i]
            best_knn_feat = features[j]

print("Best k %d, best no of features %d, kNN accuracy %g\n"%(best_k, best_knn_feat, max_knn_acc))


print("Naive Bayesian...")
NB_acc = []
for feat in features:
    # preprocess data
    trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False, feat)

    # k-Nearest Neighbour
    model = GaussianNB()
    model.fit(trainX, trainY)
    predicted = model.predict(testX)
    acc = accuracy_score(testY, predicted)
    print('No of features %d, NB accuracy %g'%(feat, acc))
    NB_acc.append(acc)

plt.figure()
plt.plot(range(len(features)), NB_acc)
plt.xticks(range(len(features)), features)
plt.xlabel('no of features')
plt.ylabel('accuracy')
plt.title("NB Accuracy against no of features")
plt.savefig('Naive Bayesian.png')

for i in range(len(features)):
    max_NB_acc = max(NB_acc)
    best_NB_feat = features[NB_acc.index(max_NB_acc)]

print("Best no of features %d, Naive Bayesian accuracy %g"%(best_NB_feat, max_NB_acc))

plt.show()