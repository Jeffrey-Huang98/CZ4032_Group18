import preprocessing
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import os


def features_removed(num_features, x, y):
    estimator = RandomForestClassifier(n_estimators=10, criterion="entropy")
    selector = RFE(estimator, num_features, step=1)
    selector = selector.fit(x, y)

    arr = selector.support_
    remove = []
    for i in range(arr.size):
        if not arr[i]:
            remove.append(x.columns[i])

    return remove


# preprocess data
trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False)

# reduce number of features to 10
remove = features_removed(10, trainX, trainY)
reducedX = trainX.drop(remove, axis=1)
reducedtest = testX.drop(remove,axis=1)

# build decision tree
dec_tree = DecisionTreeClassifier(criterion="entropy")
dec_tree = dec_tree.fit(reducedX, trainY)

test_prediction = dec_tree.predict(reducedtest)

# model Accuracy
print("Accuracy:", accuracy_score(testY, test_prediction))

# adding path
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
# visualising decision tree
dot_data = StringIO()
export_graphviz(dec_tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = reducedX.columns,class_names=['M','B'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Decision Tree.png')
Image(graph.create_png())

