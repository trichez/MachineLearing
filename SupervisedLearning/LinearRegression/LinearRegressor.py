# Linear regressor
# Obs1: as long as the data set is an CSV file, it works for any-size data sets# Author: Matheus Henrique Trichez
# License: GPL3
# param1: data set
# ----------------
# synopsis
#	python3 LinearRegressor.py FoodTruck.csv
#	python3 LinearRegressor.py prepAirQualityUCI.csv
#	python3 LinearRegressor.py wine.csv
import numpy as np
import matplotlib.pyplot as plt
import sys


def h(X, Theta):
	return X.dot(Theta.T)

def costFunction(X, y, Theta): # cost function
	m = X.shape[0] # size of dataset (i.e. number of rows)
	y_hat = X.dot(Theta.T)

	cost = np.sum(np.sqrt((y-y_hat)**2))
	return cost/(2*m)

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

# mean squared error
def mse(X, y, y_hat):
	m = X.shape[0]
	return (np.sum((y_hat - y)**2))/m

# Root mean squared error
def rmse(mse):
	return np.sqrt(mse)

lines = []
thetaP = []
XtrainLen = 0.7 # the trainning set correspond to 70% of the total amount of data
nFeatures = 0
Alpha = 0.00001 # Learning rate


fileReader = open(sys.argv[1],'r')
lines = fileReader.readlines() #fileReader is now on EOF

for c in lines[0]: #counts how many commas has the file, so while its a csv file it should work for any dimension
	if c == ',':
		nFeatures = nFeatures + 1
print("# features", nFeatures)
Theta = np.array([np.ones(nFeatures)])

y = np.array([[i.split(',')[nFeatures][:-1]] for i in lines], dtype=float)
X =	np.array([k.split(',')[0:nFeatures] for k in lines], dtype=float)
X = np.delete(X, 0, axis=1)
X = np.insert(X, 0,1, axis=1) #insert a column of 1's so we can use the bias (so the matrix mulltiplication could work)

print("\ntheta shape:", Theta.shape)
print("y shape:", y.shape)
print("X shape:", X.shape)

f=1
for f in range(nFeatures): # scailing data by some policy, changing it may improve your models performance
	maxFeature = np.max(X[:,f],0)

	if maxFeature < 1.0:
		continue
	else:
		X[:,f] = X[:,f] / maxFeature  										# scailing(i.e. normalize) the data, because of de disantce between mean values of the features

#splitting the data set into trainning and testing sets
tSize = int(np.shape(X)[0]*XtrainLen)
xTrain = X[:tSize]
xTest = X[tSize:]
yTrain = y[:tSize]
yTest = y[tSize:]


Theta, J = gradDescendent(xTrain, yTrain , Theta, Alpha, 0.001)
y_hat = h(xTest, Theta)

print("RSME: ", rmse(mse(xTest, yTest, y_hat)))
print("Cost Function on Trainning set: ", costFunction(xTrain, yTrain, Theta))
print("Cost Function on Testing set: ", costFunction(xTest, yTest, Theta))
