import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

X_train, X_test, y_train, y_test = preprocessing.preprocess(0.3, False)

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

# Build network
model = Sequential()
model.add(Dense(num_neurons, input_dim=NUM_FEATURES, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
sgd = optimizers.SGD(lr=learning_rate, decay=decay)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=epochs)