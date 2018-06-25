import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics

param_grid = {
    "min_impurity_decrease": [np.arange(0.01, 2, 0.1)],
    "presort": [True],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 9, 10],
    "min_weight_fraction_leaf" :[np.arange(0.01, 1, 0.05)]
}

dataset = datasets.load_iris()

X = dataset.data
y = dataset.target
X, y = shuffle(X, y, random_state=0)# it is that ok?

#splitting the data set into trainning and testing sets
tSize = int(np.shape(X)[0]*0.7)
xTrain = X[:tSize]
xTest = X[tSize:]
yTrain = y[:tSize]
yTest = y[tSize:]




dtClassifier = DecisionTreeClassifier(random_state=0)
GSC = GridSearchCV(dtClassifier, param_grid, scoring="accuracy", cv=5)
GSC.fit(xTrain, yTrain)
y_hat = GSC.predict(yTest)
