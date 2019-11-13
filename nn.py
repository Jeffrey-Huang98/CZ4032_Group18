import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
import pylab as plt
from keras import backend as K


# Initialize
NUM_FEATURES = 30
NUM_CLASSES = 2

learning_rate = 0.01
num_epochs = 500
batch_size = 32
num_neurons = 10
decay = 1e-6
seed = 10
np.random.seed(seed)

# preprocess data
trainX_p, testX_p, trainY_p, testY_p = preprocessing.preprocess(0.3, False)
trainY = trainY_p.replace(['B', 'M'], [1, 2])
trainX = trainX_p.values
trainY = trainY.values
testY = testY_p.replace(['B', 'M'], [1, 2])
testX = testX_p.values
testY = testY.values

# create a matrix of dimension train_y row x 2 filled with zeros
trainY_onehot = np.zeros((trainY.shape[0], NUM_CLASSES))
testY_onehot = np.zeros((testY.shape[0], NUM_CLASSES))
# create classification matrix
trainY_onehot[np.arange(trainY.shape[0]), trainY-1] = 1  # one hot matrix
testY_onehot[np.arange(testY.shape[0]), testY-1] = 1  # one hot matrix

history = []
features = [10, 15, 20, 25, 30]
for x in features:
    tmp = preprocessing.remove_features(x, trainX, trainY)
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=x, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    sgd = optimizers.SGD(lr=learning_rate, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history.append(model.fit(tmp, trainY_onehot, epochs=num_epochs))

plt.figure()
for i in range(len(features)):
    plt.plot(range(num_epochs), history[i].history['accuracy'], label=str(features[i]) + ' features')
plt.xlabel(str(num_epochs) + ' iterations')
plt.ylabel('Metrics')
plt.legend(loc='lower right')

plt.show()