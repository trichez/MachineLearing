# License: GPL3
# param1: data set
# ----------------
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.utils import shuffle
import numpy as np
import sys


k = 1
nIters = 1
iterNumber = 0
lines = []
uc = []
centroids = []
dists = []
distances = []
costs = []
XtrainLen = 0.7
nFeatures = 0
separetor = ','
np.random.seed(564)

def calcDist(u , v):

	return np.sqrt(np.sum(np.subtract(u,v)**2))

def costFunction(X, uc): # cost function
	m = X.shape[0] # size of dataset (i.e. number of rows)

	return (1/m) * np.sum(np.linalg.norm((X - uc)**2))



fileReader = open("RelationNetwork.csv",'r')
lines = fileReader.readlines() #fileReader is now on EOF
#lines = shuffle(lines, random_state=np.random.random_integers(1545))

for c in lines[0]: #counts how many commas has the file, so while its a csv file it should work for any dimension
	if c == separetor:
		nFeatures = nFeatures + 1

X =	np.array([k.split(separetor)[0:nFeatures] for k in lines], dtype=float)

while k <= 15:
	print("computing k equals to:", k)
	centroids[:] = []
	iterNumber = 0
	for centrId in range(k):
		 centroids.append(X[np.random.random_integers(X.shape[0])]) #random samples as initial centroids

	while iterNumber < nIters: #
		distances[:] = k*[[]] # matrix ... for centroid recalculation
		uc[:] = []
		for xi in range(X.shape[0]):
			dists[:] = []

			for centrId in range(k):
				dists.append(calcDist(centroids[centrId], X[xi])) #each position is the distance between the current tuple (X[xi]) and the centroids(identified by the indexes)

			distances.insert(np.argmin(dists),[X[xi]])
			uc.append(centroids[np.argmin(dists)])



		''' update centroids '''
		for ks in range(k):#ks is the centroid's id
			for f in range(nFeatures): # for all features of ..
				val = 0
				for xs in range(len(distances[0])): # .. each row that was assigned to this ks centroid
					 val = val + distances[ks][xs][f]
				centroids[ks][f] = val /  len(distances[0])

		costs.append(costFunction(X,uc))
		iterNumber += 1

	k += 1

print("Best K: ", np.argmin(costs))
plt.figure()
plt.xlabel("number of clusters")
plt.ylabel("Cost Function")
plt.title("close this window to continue")
plt.plot(np.arange(1,16, 1), costs)
plt.grid(True)
plt.show()
plt.close()

k = np.argmin(costs)
centroids[:] = []
iterNumber = 0
costs[:] = []

for centrId in range(k):
	 centroids.append(X[np.random.random_integers(X.shape[0])]) #random samples as initial centroids
print("running 100 times with the new K... (that can take some time)")
while iterNumber < 100: #
	distances[:] = k*[[]] # matrix ... for centroid recalculation
	uc[:] = []
	for xi in range(X.shape[0]):
		dists[:] = []

		for centrId in range(k):
			dists.append(calcDist(centroids[centrId], X[xi])) #each position is the distance between the current tuple (X[xi]) and the centroids(identified by the indexes)

		distances.insert(np.argmin(dists),[X[xi]])
		uc.append(centroids[np.argmin(dists)])



	''' update centroids '''
	for ks in range(k):#ks is the centroid's id
		for f in range(nFeatures): # for all features of ..
			val = 0
			for xs in range(len(distances[0])): # .. each row that was assigned to this ks centroid
				 val = val + distances[ks][xs][f]
			centroids[ks][f] = val /  len(distances[0])

	costs.append(costFunction(X,uc))

	iterNumber += 1

print("iteratration number {} was the best".format(np.argmin(costs)))
