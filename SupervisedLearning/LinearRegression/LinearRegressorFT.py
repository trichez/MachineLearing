import numpy as np
import matplotlib.pyplot as plt

lines = []
Theta = np.array([[0, 0.15]])
XtrainLen = 0.7

def costFunction(X, y, Theta): # cost function
	m = X.shape[0] # size of dataset (i.e. number of rows)
	y_hat = X.dot(Theta.T)

	cost = np.sum(np.sqrt((y-y_hat)**2))
	return cost/(2*m)

def gradDescendent(X, y, Theta, alpha, itNumber):
	m = X.shape[0]
	J = []

	for i in range(itNumber):
		y_hat = X.dot(Theta.T)
		error = (y_hat - y) * X
		error = np.sum(error,0)/m
		Theta = Theta - (alpha * error)
		J.append(costFunction(X,y,Theta))
	
	return Theta, J	
	 #error = 1,d+1 // shape
def h(X, Theta):
	return X.dot(Theta.T)
		 
fileReader = open("FoodTruck.csv",'r')
lines = fileReader.readlines() #fileReader is now on EOF


y = np.array([[i.split(',')[1][:-1]] for i in lines], dtype=float)
X =	np.array([i.split(',')[0:1] for i in lines],dtype=float)
X = np.insert(X, 0,1, axis=1) #insert a column of 1's

print("\ntheta shape:", Theta.shape)
print("y shape:", y.shape)
print("X shape:", X.shape)

tSize = int(np.shape(X)[0]*XtrainLen)
xTrain = X[:tSize]
xTest = X[tSize:]
yTrain = y[:tSize]
yTest = y[tSize:]


Theta, J = gradDescendent(xTrain, yTrain , Theta, 0.02, 1000)


#y_hat = h(xTrain, Theta)
plt.figure(1)

#plt.plot(xTrain[:,1],yTrain, 'b*')
#plt.plot(xTrain[:,1],y_hat,'r-^')
print("Cost Function on Trainning set: ", costFunction(xTrain, yTrain, Theta))

y_hat = h(xTest, Theta)

plt.plot(xTest[:,1],yTest, 'b*')
plt.plot(xTest[:,1],y_hat,'r-^')

print("Cost Function on Testing set: ", costFunction(xTest, yTest, Theta))
print("Max value:", np.max(y_hat))
print("Min value:", np.min(y_hat))

plt.show()
