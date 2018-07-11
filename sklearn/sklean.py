import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

param_gridDT = {
    "min_impurity_decrease": [0.1, 0.3, 0.5, 0.7, 0.9, 0.11, 0.13, 0.15, 0.17, 0.19],
    "presort": [True],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 9, 10],
    "min_weight_fraction_leaf" :[0.0 ,0.1, 0.2, 0.4, 0.5]
}
param_gridLR = {
    "solver": [ 'newton-cg'],
    "multi_class": ['multinomial'],
    "tol": [0.001, 0.0001, 0.00001]
}
param_gridPA = {
    "C": [0.5, 1 ,1.5],
    "tol": [0.001, 0.0001, 0.00001],
    "warm_start": [True],
    "average ": [True]
}
lines = []
nFeatures = 0
separator = ','


fileReader = open("diabetic_data.csv",'r')
lines = fileReader.readlines() #fileReader is now on EOF
lines = lines[1:]
#lines = shuffle(lines, random_state=np.random.random_integers(568))
for c in lines[0]: #counts how many commas has the file, so while its a csv file it should work for any dimension
	if c == separator:
		nFeatures = nFeatures + 1

y = np.array([[i.split(separator)[nFeatures]] for i in lines])
X =	np.array([k.split(separator)[0:nFeatures] for k in lines])
#splitting the data set into trainning and testing sets
print("\npreprocessing the dataset...")

X = np.delete(X, 0, 1)  # (self, column, axis)
X = np.delete(X, 0, 1)  # (self, column, axis)
X = np.delete(X, 3, 1)  # (self, column, axis)
X = np.delete(X, 5, 1)  # (self, column, axis)
X = np.delete(X, 5, 1)  # (self, column, axis)
X = np.delete(X, 5, 1)  # (self, column, axis)
X = np.delete(X, 5, 1)  # (self, column, axis)
X = np.delete(X, 8, 1)  # (self, column, axis)
X = np.delete(X, 8, 1)  # (self, column, axis)
X = np.delete(X, 8, 1)  # (self, column, axis)
X = np.delete(X, 12, 1)  # (self, column, axis)
X = np.delete(X, 12, 1)  # (self, column, axis)
for i in range (0,17):
    X = np.delete(X, 12, 1)  # (self, column, axis)
X = np.delete(X, 13, 1)  # (self, column, axis)
X = np.delete(X, 13, 1)  # (self, column, axis)
X = np.delete(X, 13, 1)  # (self, column, axis)
X = np.delete(X, 13, 1)  # (self, column, axis)
X = np.delete(X, 13, 1)  # (self, column, axis)
X = np.delete(X, 0, 1)  # (self, column, axis)
#X = np.delete(X, 13, 1)  # (self, column, axis)
X = np.delete(X, 7, 1)  # (self, column, axis)
X = np.delete(X, 7, 1)  # (self, column, axis)
X = np.delete(X, 7, 1)  # (self, column, axis)
X = np.delete(X, 0, 1)  # (self, column, axis)

rms = 0
for i in range(0, X.shape[0]):
    for j in range(0, X.shape[1]):

        if X[i][j] == 'Female':
            X[i][j] = 0.
        elif X[i][j] == 'Male':
            X[i][j] = 1.

        if X[i][j][0] == '[':
            str = X[i][j].split('[')
            str = str[1].split('-')
            X[i][j] =  (int(str[0]) + int(str[1][:-1]) ) / 2

        if X[i][j] == 'No':
            X[i][j] = 0.
        elif X[i][j] == 'Ch':
            X[i][j] = 1.
        elif X[i][j] == 'Yes':
            X[i][j] = 1.
        elif X[i][j] == 'Up':
            X[i][j] = 1.
        elif X[i][j] == 'Steady':
            X[i][j] = 2.
        elif X[i][j] == 'Down':
            X[i][j] = 3.


        if X[i][j] == '?' or X[i][j] == 'Unknown/Invalid':
            X = np.delete(X, i-rms, 0)
            rms = rms + 1


X = X.astype(float)

tSize = int(np.shape(X)[0]*0.9)
xTrain = X[:tSize]
xTest = X[tSize:]
yTrain = y[:tSize]
yTest = y[tSize:]
print("\tDONE. (the next stages might take a while)\n")


print("\nRunning Decision Tree Classifier...")
dtClassifier = DecisionTreeClassifier(random_state=0)
GSC = GridSearchCV(dtClassifier, param_gridDT, scoring="accuracy", cv=5)
GSC.fit(xTrain, yTrain)
y_hat = GSC.predict(xTest)
print("Decision Tree Report:")
print(metrics.classification_report(yTest, y_hat))


print("\nLogistic Regression Classifier...")
logistic = linear_model.LogisticRegression()
GSC = GridSearchCV(logistic, param_gridLR, scoring="accuracy", cv=5)
GSC.fit(xTrain, yTrain.ravel())
y_hat = GSC.predict(xTest)
print("Logistic Regression Report: ")
print(metrics.classification_report(yTest, y_hat))
