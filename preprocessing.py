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


def remove_features(num_features, x, y):
    estimator = SVR(kernel='linear')
    selector = RFE(estimator, num_features, step=1)
    selector = selector.fit(x, y)
    arr = selector.support_
    remove = []
    for i in range(arr.size):
        if not arr[i]:
            remove.append(i)
    return np.delete(x, remove, 1)

def preprocess(test_size, display):
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

    # Checking for categoriacal data
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

    # Display diagrams
    if display:
        plt.show()

    return X_train, X_test, y_train, y_test