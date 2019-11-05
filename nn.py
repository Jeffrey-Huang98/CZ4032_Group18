import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
import pylab as plt
from keras import backend as K
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def remove_features(num_features, x, y):
    estimator = SVR(kernel='linear')
    selector = RFE(estimator, num_features, step=1)
    selector = selector.fit(x, y)
    arr = selector.support_
    remove = []
    for i in range(arr.size):
        if not arr[i]:
            remove.append(i)
    return np.delete(X_train, remove, 1)

# Initialize
NUM_FEATURES = 30
NUM_CLASSES = 2

learning_rate = 0.01
num_epochs = 50
batch_size = 32
num_neurons = 10
decay = 1e-6
seed = 10
epochs = 50
np.random.seed(seed)

X_train, X_test, trainY, testY = preprocessing.preprocess(0.3, False)
trainY = trainY.replace(['B', 'M'], [1, 2])
X_train = X_train.values
trainY = trainY.values
testY = testY.replace(['B', 'M'], [1, 2])
X_test = X_test.values
testY = testY.values

# create a matrix of dimension train_y row x 2 filled with zeros
y_train = np.zeros((trainY.shape[0], NUM_CLASSES))
y_test = np.zeros((testY.shape[0], NUM_CLASSES))
# create classification matrix
y_train[np.arange(trainY.shape[0]), trainY-1] = 1  # one hot matrix
y_test[np.arange(testY.shape[0]), testY-1] = 1  # one hot matrix

# Build network
model = Sequential()
model.add(Dense(num_neurons, input_dim=NUM_FEATURES, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
sgd = optimizers.SGD(lr=learning_rate, decay=decay)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy', precision_m])
history = model.fit(X_train, y_train, epochs=num_epochs)

plt.figure(1)
plt.plot(range(epochs), history.history['accuracy'], label='Accuracy')
plt.plot(range(epochs), history.history['precision_m'], label='Precision')
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Metrics')
plt.legend(loc='lower right')

plt.show()