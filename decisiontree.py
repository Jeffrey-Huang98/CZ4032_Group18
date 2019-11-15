import utils
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import os
import pylab as plt
import time


# preprocess data
trainX_, testX_, trainY_, testY_ = utils.preprocess(0.3, False)

features = [10, 15, 20, 25, 30]
dectree_accuracy = []

best_acc = 0
for feat in features:
    # reduce feature
    trainX, testX, trainY, testY = utils.remove_features(feat, trainX_, testX_, trainY_, testY_)

    # build decision tree
    dec_tree = DecisionTreeClassifier(criterion="entropy")
    dec_tree = dec_tree.fit(trainX, trainY)

    predicted = dec_tree.predict(testX)

    # model Accuracy
    acc = accuracy_score(testY, predicted)
    dectree_accuracy.append(acc)
    print("No of features %d accuracy %g"%(feat, acc))

    if acc > best_acc:
        best_feat = feat
        best_acc = acc


plt.figure()
plt.plot(range(len(features)), dectree_accuracy)
plt.xticks(range(len(features)), features)
plt.xlabel('no of features')
plt.ylabel('accuracy')
plt.title("Accuracy against no of features")
plt.savefig('img/dectree.png')

print("Decision tree accuracy:", dectree_accuracy)
print("Best no of features %d accuracy: %g"%(best_feat, best_acc))


# optimal decision tree
start = time.time()
# reduce feature
trainX, testX, trainY, testY = utils.remove_features(best_feat, trainX_, testX_, trainY_, testY_)

# build decision tree
dec_tree = DecisionTreeClassifier(criterion="entropy")
dec_tree = dec_tree.fit(trainX, trainY)
predicted = dec_tree.predict(testX)

# model Accuracy
acc = accuracy_score(testY, predicted)
dectree_runtime = time.time() - start
print('Optimal decision tree runtime: ', dectree_runtime)




# adding path
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
# visualising decision tree
dot_data = StringIO()
export_graphviz(dec_tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = trainX.columns,class_names=['M','B'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('img/Decision Tree.png')
Image(graph.create_png())

plt.show()
