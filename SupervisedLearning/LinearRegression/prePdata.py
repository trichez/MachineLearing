import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd


fileReader = open('/home/trichez/theniosclass/AirQualityUCI/AirQualityUCI.csv')

data = np.array([l.split(';') for l in fileReader.readlines()])
data = data[1:,:] # handler
data = data[:,:-2] # last 2 columns (empty)
data = data[:,2:] # date and time (first wo columns)
#print(np.nonzero(data[:,0]==''))
#x = numpy.delete(data, (0), axis=0)
#deleting empty lines
data = np.delete(data, [np.nonzero(data[:,0]=='')], axis=0)# axis=0 means that you are selecting rows
#now delete lines with some with no value '-200'
data = np.delete(data, [np.nonzero(data[:,:]=='-200')], axis=0)# axis=0 means that you are selecting rows

data[:,0] = [float(i[0].replace(',','.')) for i in data] # the symbol used in float is a dot, not a comma
data[:,3] = [float(j[3].replace(',','.')) for j in data]
data[:,10] = [float(k[10].replace(',','.')) for k in data]
data[:,11] = [float(l[11].replace(',','.')) for l in data]
data[:,12] = [float(m[12].replace(',','.')) for m in data]

y = np.array([[v] for v in data[:,3]]) # our y
data = np.delete(data, 3, axis=1) # remove y from the middle of the matrix

data = np.append(data, y, axis=1) # insert y as the last column of the matrix (because its seems to be kinda standard good practice )

df = pd.DataFrame(data) # parse the numpy array to save it as a csv file

df.to_csv("prepAirQualityUCI.csv",index=False) # WARNNING: panda put in a index columns
