import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
import pylab as plt

X_train, X_test, trainY, testY = preprocessing.preprocess(0.3, False)
trainY = trainY.replace(['B','M'],[1,2])
X_train = X_train.values
trainY = trainY.values

# Initialize
NUM_FEATURES = 30
NUM_CLASSES = 2

learning_rate = 0.01
epochs = 500
batch_size = 32
num_neurons = 10
decay = 1e-6
seed = 10
epochs = 50
np.random.seed(seed)

# create a matrix of dimension train_y row x 2 filled with zeros 
y_train = np.zeros((trainY.shape[0], NUM_CLASSES))
# create classification matrix
y_train[np.arange(trainY.shape[0]), trainY-1] = 1 #one hot matrix

# Build network
model = Sequential()
model.add(Dense(num_neurons, input_dim=NUM_FEATURES, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
sgd = optimizers.SGD(lr=learning_rate, decay=decay)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs)

plt.figure(1)
plt.plot(range(epochs), history.history['accuracy'])
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()