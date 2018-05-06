import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import sys
import os       
os.environ["PATH"] += os.pathsep + 'D:/gra/bin'

iris = load_iris()
x=[]
filename = 'iris.txt'
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()
        if not lines:
            break
        tmp0, tmp1, tmp2, tmp3,temp = [i for i in lines.split(",")]
        a=[]
        a.append(float(tmp0))
        a.append(float(tmp1))
        a.append(float(tmp2))
        a.append(float(tmp3))
        x.append(a)
data=np.array(x)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, iris.target)
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")