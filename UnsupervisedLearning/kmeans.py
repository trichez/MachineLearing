# License: GPL3
# param1: data set
# ----------------
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.utils import shuffle
import numpy as np
import sys


k = 2
lines = []
thetaP = []
XtrainLen = 0.7
nFeatures = 0
Alpha = 0.01 # Learning rate
Epsilon = 0.000001
separetor = ','
#np.random.seed(56484)

fileReader = open("Admission.txt",'r')
lines = fileReader.readlines() #fileReader is now on EOF

lines = shuffle(lines, random_state=np.random.random_integers(1545))

for c in lines[0]: #counts how many commas has the file, so while its a csv file it should work for any dimension
	if c == separetor:
		nFeatures = nFeatures + 1

print("# features", nFeatures)
y = np.array([[i.split(separetor)[nFeatures][:-1]] for i in lines], dtype=int)
X =	np.array([k.split(separetor)[0:nFeatures] for k in lines], dtype=float)

print("y shape:", y.shape)
print("X shape:", X.shape)


center1 = np.array( [float(X[:,0].argmin()), float(X[:,1].argmin())] )
center0 = np.array( [float(X[:,0].argmax()), float(X[:,1].argmax())] )


k = 1
while True:
	if k > 4:
		break
	plt.subplot(2,2,k)
	plt.title(k)
	plt.scatter(X[:,0], X[:,1])
	plt.scatter((center0[0],center1[0]), (center0[1], center1[1]), color='r')
	plt.grid(True)

	print(center0)
	print(center1)
	print("\n")
	k = k + 1

	for tupl in range(0,X.shape[0]):
		if distance.euclidean(center0,tupl) > distance.euclidean(center1,tupl):
				center0[0] = (center0[0] + X[tupl][0])/2
				center0[1] = (center0[1] + X[tupl][1])/2
		else:
			center1[0] = (center1[0] + X[tupl][0])/2
			center1[1] = (center1[1] + X[tupl][1])/2


plt.show()
