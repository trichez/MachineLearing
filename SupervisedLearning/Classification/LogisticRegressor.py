# Logistic regressor based on Linear Regressor
# Obs1: uses sigmoid function to transform the linear regressor into a logistic regressor
# Obs2: as long as the data set is an CSV file, it works for any-size data sets
# Author: Matheus Henrique Trichez
# License: GPL3
# param1: data set
# ----------------
# synopsis:
# python3 LogisticRegressor.py sudents.csv
import numpy as np
import matplotlib.pyplot as plt
import sys

def sigmoid(z):
	return 1 / (1+ np.e**(-z))

def h(X, Theta):
	return X.dot(Theta.T)

def costFunction(X, y, Theta): # cost function
	m = X.shape[0] # size of dataset (i.e. number of rows)
	y_hat = sigmoid(h(X,Theta))

	return (-y * np.log(y_hat) - (1-y) * np.log(y_hat)).sum(0)/m

def gradDescendent(X, y, Theta, alpha, epsilon):
	m = X.shape[0]
	J = []
	lastScalarError = 1
	scalarError = 0

	while abs(scalarError - lastScalarError) > epsilon:
		lastScalarError = scalarError

		y_hat = h(X,Theta)
		error = (y_hat - y) * X
		error = np.sum(error,0)/m
		Theta = Theta - (alpha * error)
		scalarError = costFunction(X,y,Theta)
		J.append(scalarError)

	return Theta, J


def mse(X, y, y_hat):

	m = X.shape[0]
	return (np.sum((y_hat - y)**2))/m

def rmse(mse):

	return np.sqrt(mse)

lines = []
thetaP = []
XtrainLen = 0.7
nFeatures = 0
Alpha = 0.0001 # Learning rate



fileReader = open(sys.argv[1],'r')
lines = fileReader.readlines() #fileReader is now on EOF

for c in lines[0]: #counts how many commas has the file, so while its a csv file it should work for any dimension
	if c == ',':
		nFeatures = nFeatures + 1
Theta = np.array([np.ones(nFeatures+1)])
print("# features", nFeatures)

y = np.array([[i.split(',')[nFeatures][:-1]] for i in lines], dtype=float)
X =	np.array([k.split(',')[0:nFeatures] for k in lines], dtype=float)
X = np.insert(X, 0,1, axis=1) #insert a column of 1's so we can use the bias

print("theta shape:", Theta.shape)
print("y shape:", y.shape)
print("X shape:", X.shape)

'''
for scailing features, if needed
f=1
for f in range(nFeatures):  # scailing data by some policy, changing it may improve your models performance

	maxFeature = np.max(X[:,f],0)

	if maxFeature < 1.0:
		continue
	else:
		X[:,f] = X[:,f] / maxFeature  										# scailing(i.e. normalize) the data (because of de disantce between mean values of the features)

'''
tSize = int(np.shape(X)[0]*XtrainLen)
xTrain = X[:tSize]
xTest = X[tSize:]
yTrain = y[:tSize]
yTest = y[tSize:]


Theta, J = gradDescendent(xTrain, yTrain , Theta, Alpha, 0.000001)
y_hat = np.round(sigmoid(h(xTest, Theta)))

print("Accuracy: ", np.mean(yTest == y_hat)*100,'%')
