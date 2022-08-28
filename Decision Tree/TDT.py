import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("Iterations.csv")

print(df)

translations = {"CLASS" : 0, "EAT" : 1, "NONE" : 2, "REST" : 3, "STUDY": 4}

df["CURRENTACTION"] = df["CURRENTACTION"].map(translations)
df["PREVIOUSACTION"] = df["PREVIOUSACTION"].map(translations) 
df["NEXTSCHEDULE"] = df["NEXTSCHEDULE"].map(translations) 
df["RESULT"] = df["RESULT"].map(translations) 

print(df)

df.to_csv("Treated.csv")

features = ["CURRENTACTION","PREVIOUSACTION","NEXTSCHEDULE","TIREDNESSINDEX"]

X = df[features]
y = df["RESULT"]

print("Features:")
print(X)

print("End Results")
print(y)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X,y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('ThesisDecisionTree.png')

img=pltimg.imread('ThesisDecisionTree.png')
imgplot = plt.imshow(img)
plt.show()