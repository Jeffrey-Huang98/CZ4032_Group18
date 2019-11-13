# Import library
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
plt.style.use('ggplot')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier


def remove_features(num_features, x, y):
    estimator = RandomForestClassifier(n_estimators=10, criterion="entropy")
    selector = RFE(estimator, num_features, step=1)
    selector = selector.fit(x, y)
    arr = selector.support_
    remove = []
    for i in range(arr.size):
        if not arr[i]:
            remove.append(x.columns[i])
    return remove

def preprocess(test_size, display, num_features):
    # Read data
    df = pd.read_csv('data.csv')
    df.head()
    if display:
        print("Dataset dimension:")
        print(df.shape)

        # Checking for missing values
        print("Number of missing values:")
        print(df.isnull().any().sum())

        # Remove empty column
        print('Remove empty column...')
    df = df.dropna(axis=1,how='all')
    
    if display:
        print("Dataset dimension:")
        print(df.shape)

        # Checking for missing values
        print("Number of missing values:")
        print(df.isnull().any().sum())

    All = df.shape[0]
    benign = df[df['diagnosis'] == 'B']
    malignant = df[df['diagnosis'] == 'M']

    if display:
        print('Malignant:', len(benign)/All*100, '%')
        print('Benign:', len(malignant)/All*100, '%')

        plt.figure(1)
        labels = ['benign','malignant']
        classes = pd.value_counts(df['diagnosis'], sort = True)
        classes.plot(kind = 'bar', rot=0)
        plt.title("Diagnosis distribution")
        plt.xticks(range(2), labels)
        plt.xlabel("Diagnosis")
        plt.ylabel("Frequency")

    # Checking for categorical data
    correlation_matrix = df.corr()
    if display:
        fig = plt.figure(2, figsize=(12,9))
        sns.heatmap(correlation_matrix,vmax=0.8,square = True)

    # Standardize data
    for key in df.keys():
        if key != 'id' and key != 'diagnosis':
            df[key] = StandardScaler().fit_transform(df[key].values.reshape(-1,1))
    x = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    if display:
        print(x)
        print(y)

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=10)

    # reduce number of features to 10
    remove = remove_features(num_features, X_train, y_train)
    reduced_X_train = X_train.drop(remove, axis=1)
    reduced_X_test = X_test.drop(remove,axis=1)
    
    correlation_matrix = reduced_X_train.corr()
    if display:
        fig = plt.figure(3, figsize=(12,9))
        sns.heatmap(correlation_matrix,vmax=0.8,square = True)

        plt.show()

    
    return reduced_X_train, reduced_X_test, y_train, y_test