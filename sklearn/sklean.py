import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics

param_grid = {#np.arange(0.01, 2, 0.1)
    "min_impurity_decrease": [0.1, 0.3, 0.5, 0.7, 0.9, 0.11, 0.13, 0.15, 0.17, 0.19],
    "presort": [True],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 9, 10],
    "min_weight_fraction_leaf" :[0.0 ,0.1, 0.2, 0.4, 0.5]
}
'''
dataset = datasets.load_iris()

X = dataset.data
y = dataset.target
X, y = shuffle(X, y, random_state=0)# it is that ok?
'''

lines = []
thetaP = []
XtrainLen = 0.7
nFeatures = 0
Alpha = 0.01 # Learning rate
Epsilon = 0.000001
separator = ','


fileReader = open("diabetic_data.csv",'r')
lines = fileReader.readlines() #fileReader is now on EOF
#lines = shuffle(lines, random_state=np.random.random_integers(1545))

for c in lines[0]: #counts how many commas has the file, so while its a csv file it should work for any dimension
	if c == separator:
		nFeatures = nFeatures + 1

y = np.array([[i.split(separator)[nFeatures]] for i in lines])
X =	np.array([k.split(separator)[0:nFeatures] for k in lines])
#splitting the data set into trainning and testing sets
X = X[1:]
y = y[1:]

print(y)
'''
tSize = int(np.shape(X)[0]*0.7)
xTrain = X[:tSize]
xTest = X[tSize:]
yTrain = y[:tSize]
yTest = y[tSize:]


dtClassifier = DecisionTreeClassifier(random_state=0)
GSC = GridSearchCV(dtClassifier, param_grid, scoring="accuracy", cv=5)
GSC.fit(xTrain, yTrain)
y_hat = GSC.predict(xTest)
print(metrics.classification_report(yTest, y_hat))
'''
