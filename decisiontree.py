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

# preprocess data
trainX, testX, trainY, testY = preprocessing.preprocess(0.3, False, 10)

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

