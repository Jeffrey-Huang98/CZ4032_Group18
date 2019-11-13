# Breast Cancer Prediction
The aim of this project is to predict whether a breast tumor is benign or malignant by means of various data mining classification techniques.

## Group
CZ4032 Group 18
1. Gantari Evanda Raufani
2. Jeffrey
3. Michelle Vanessa
4. Natassa Karinka
5. Stella Marcella
6. Wilbert

## Requirements
The code for this project is written in [Python](https://www.python.org/). The code needs to be run inside a virtual environment to isolate package installation from the system. To use a virtual environment, run the following command on the terminal:
```bash
# create virtual environment (this creates a virtual environment directory called venv)
$ virtualenv --system-site-packages -p python ./venv 

# activate the virtual environment
$ source ./venv/bin/activate    # on ubuntu or macOS
$ .\venv\Scripts\activate       # on windows

# to exit the virtual environment
(venv) $ deactivate
```

### Setup
All this dependencies needs to be installed using [pip](https://pypi.org/project/pip/) before running the program.
1. [Pandas](https://pandas.pydata.org/)
2. [Numpy](https://numpy.org/)
3. [Matplotlib](https://matplotlib.org/)
4. [Seaborn](https://seaborn.pydata.org/)
5. [Scikit-learn](https://scikit-learn.org/stable/)
6. [Keras](https://keras.io/)
7. [Tensorflow](https://www.tensorflow.org/)

## Run Program
The codes for the algorithms are written in separate files. Run the following commands from the root of the repository:
```bash
# Artificial Neural Networks
$ python nn.py

# Decision Tree
$ python decisiontree.py

# KNN and Naive Bayesian
$ python knnbayesian.py

# K-Means Clustering
$ python clustering.py
```