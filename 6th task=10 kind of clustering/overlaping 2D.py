# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:05:47 2022

@author: Foroogh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:35:52 2022

@author: Foroogh
"""

import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import OPTICS

"""
 ---- Loading data ----
"""
file = pd.read_csv('Signalnumber2.csv' ,skiprows=0)
#aaa=file[0].to_csv()
data = file.to_numpy()
#divData=(data).reshape(189,100) 
#Data index 0 signal with noise , index 1 signal without noise 


# overlaping 
datapoint=0
first=50
end=50
windowsize=100 # size of window
numberOfSeg=int((16383/(first))-(end/first))
divData=np.zeros((numberOfSeg,windowsize));
for i in range(0, numberOfSeg):
   if(i!=0):
       datapoint=datapoint-end
   else:
       datapoint=0
   for j in range(0, windowsize):
       datapoint+= 1
       #print(' i=',i,' j=',j,' k=',k)
       divData[i][j]=data[datapoint]


"""
 ---- Define features ----
"""
stdDD=np.ones(numberOfSeg)
meanDD=np.ones(numberOfSeg)
maxDD=np.ones(numberOfSeg)
medianDD=np.ones(numberOfSeg)
rmsDD=np.ones(numberOfSeg)
kurDD=np.ones(numberOfSeg)
skewDD=np.ones(numberOfSeg)
indexDD=np.ones(numberOfSeg)
for i in range(divData.shape[0]):
    indexDD[i]=i;
    meanDD[i]=divData[i].mean()
    maxDD[i]=max(divData[i])
    medianDD[i]=np.median(divData[i])
    rmsDD[i]=np.sqrt(np.mean(divData[i]**2))
    kurDD[i]=kurtosis(divData[i], fisher=False)
    skewDD[i]=skew(divData[i])
    stdDD[i]=np.std(divData[i])


"""
 #---- Merge features ----
"""

outputC=np.zeros((numberOfSeg,2));

mergeFtrsC=np.zeros((numberOfSeg,5)) #empty matrix
mergeFtrsR=np.vstack([indexDD, stdDD,meanDD,rmsDD]) 
for i in range(len(mergeFtrsR)):
   for j in range(len(mergeFtrsR[0])):
       mergeFtrsC[j][i] = mergeFtrsR[i][j]
np.savetxt("ftrsC.csv", mergeFtrsC, delimiter=",")
featuresX=1
featuresY=2
string1='max'
string2='rms'

"""
 #---- Shuffle ----

"""
per_list = np.random.permutation(len(mergeFtrsC))
sh_inputs = []
for i in range(len(per_list)):
    temp = per_list[i]
    tmp_inputs = mergeFtrsC[temp]
    sh_inputs.append(tmp_inputs)
sh_inputs = np.array(sh_inputs)  # شافل ورودی

"""
 #---- Train Test ----
"""

split_border = int(0.5 * len(sh_inputs)) 

x_train=sh_inputs[0:split_border, :] 

x_val= sh_inputs[split_border: , :]

x_train=x_train[x_train[:, 0].argsort()] 
x_val=x_val[x_val[:, 0].argsort()]
k=4;
#-------------------------------------------------------------
# spectral clustering
# define the model
model = SpectralClustering(n_clusters=k)
# fit model and predict clusters
yhat1 = model.fit_predict(x_val[:,1:])
plot1 = pyplot. figure(1)
pyplot.scatter(x_val[:, featuresX], x_val[:, featuresY],c=yhat1)
# show the plot
pyplot.show()
pyplot.title('spectral clustering')
pyplot.xlabel(string1)
pyplot.ylabel(string2)

#score UNsupervised learning
# Calculate Silhoutte Score
score =silhouette_score(x_val[:,1:], yhat1)
print('spectral clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#----------------------------------------------------
# birch clustering
# define the model
model = Birch(threshold=0.01, n_clusters=k)
# fit the model
model.fit(x_train[:,1:])
# assign a cluster to each example
yhat2 = model.predict(x_val[:,1:])
plot2 = pyplot. figure(2)
plt.scatter(x_val[:, featuresX], x_val[:, featuresY],c=yhat2)
# show the plot
plt.title('birch clustering')
plt.xlabel(string1)
plt.ylabel(string2)
# show the plot
plt.show()
score =silhouette_score(x_val[:,1:], yhat2)
print('birch clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#----------------------------------------------------
# agglomerative clustering

# define the model
model = AgglomerativeClustering(n_clusters=k)
# fit model and predict clusters
yhat3 = model.fit_predict(x_val[:,1:])

plot3 = pyplot. figure(3)

pyplot.scatter(x_val[:, featuresX], x_val[:, featuresY],c=yhat3)
# show the plot
pyplot.show()
pyplot.title('agglomerative clustering')
pyplot.xlabel(string1)
pyplot.ylabel(string2)
score =silhouette_score(x_val[:,1:], yhat3)
print('agglomerative clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#----------------------------------------------------
# affinity propagation clustering
# define the model
model = AffinityPropagation(damping=0.5,random_state=10)
# fit the model
model.fit(x_train[:,1:])
# assign a cluster to each example
yhat4 = model.predict(x_val[:,1:])
plot4 = pyplot. figure(4)
pyplot.scatter(x_val[:,featuresX],x_val[:,featuresY] , c=yhat4)
# show the plot
pyplot.show()
pyplot.title('affinity propagation clustering')
pyplot.xlabel(string1)
pyplot.ylabel(string2)
score =silhouette_score(x_val[:,1:], yhat4)
print('affinity propagation clustering Score=%.3f' % (score*100), '%' )
#---------------------------------------------------------------

# dbscan clustering
model = DBSCAN(eps=0.10, min_samples=9)
model.fit(x_train[:,1:])
#model.fit_predict()
yhat5=model.fit_predict(x_val[:,1:])
plot5 = pyplot. figure(5)
pyplot.scatter(x_val[:, featuresX], x_val[:, featuresY],c=yhat4)
# show the plot
pyplot.show()
pyplot.title('dbscan clustering')
pyplot.xlabel(string1)
pyplot.ylabel(string2)
score =silhouette_score(x_val[:,1:], yhat5)
print('DB scan clustering Score=%.3f' % (score*100), '%' )

#---------------------------------------------------------------
# k-means clustering
# define the model
model = KMeans(n_clusters=k)
# fit the model
model.fit(x_train[:,1:])
# assign a cluster to each example
yhat6 = model.predict(x_val[:,1:])
plot6 = pyplot. figure(6)

pyplot.scatter(x_val[:, featuresX], x_val[:, featuresY],c=yhat6)
# show the plot
pyplot.show()
pyplot.title('k-means clustering')
pyplot.xlabel(string1)
pyplot.ylabel(string2)
score =silhouette_score(x_val[:,1:], yhat6)
print('k-means clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#---------------------------------------------------------------
# mini-batch k-means clustering
model = MiniBatchKMeans(n_clusters=k)

# fit the model
model.fit(x_train[:,1:])
# assign a cluster to each example
yhat7 = model.predict(x_val[:,1:])
plot7 = pyplot. figure(7)
pyplot.scatter(x_val[:, featuresX], x_val[:, featuresY],c=yhat7)
# show the plot
pyplot.show()
pyplot.title('mini-batch k-means clustering')
pyplot.xlabel(string1)
pyplot.ylabel(string2)
score =silhouette_score(x_val[:,1:], yhat7)
print('mini-batch k-means clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#---------------------------------------------------------------
# optics clustering
# define the model
model = OPTICS(eps=0.8, min_samples=10)
# fit model and predict clusters
model.fit(x_train[:,1:])
yhat8 = model.fit_predict(x_val[:,1:])
plot8 = pyplot. figure(8)
pyplot.scatter(x_val[:, featuresX], x_val[:, featuresY],c=yhat8)
# show the plot
pyplot.show()
pyplot.title('optics clustering')
pyplot.xlabel(string1)
pyplot.ylabel(string2)
#
score =silhouette_score(x_val[:,1:], yhat8)
print(' optics clustering Score=%.3f' % (score*100), '%' )
#----------------------------------------------------------------------------
#mean shift clustering
# define the model
model = MeanShift()
# fit model and predict clusters
model.fit(x_train[:,1:])
yhat9 = model.fit_predict(x_val[:,1:])
plot9 = pyplot. figure(9)
pyplot.scatter(x_val[:, featuresX], x_val[:, featuresY],c=yhat9)
pyplot.title('mean shift clustering')
pyplot.xlabel(string1)
pyplot.ylabel(string2)

# show the plot
pyplot.show()
score =silhouette_score(x_val[:,1:], yhat9)
print('mean shift clustering Score=%.3f' % (score*100), '%' )
#----------------------------------------------------------------------------
# gaussian mixture clustering
model = GaussianMixture(n_components=4)
# fit the model
model.fit(x_train[:,1:])
# assign a cluster to each example
yhat10 = model.predict(x_val[:,1:])
plot10 = pyplot. figure(10)
pyplot.scatter(x_val[:, featuresX], x_val[:, featuresY],c=yhat10)
# show the plot
pyplot.show()
pyplot.title('gaussian mixture clustering')
pyplot.xlabel(string1)
pyplot.ylabel(string2)

# show the plot
pyplot.show()
score =silhouette_score(x_val[:,1:], yhat10)
print('gaussian mixture clustering Score=%.3f' % (score*100), '%' )
#----------------------------------------------------------------------------

