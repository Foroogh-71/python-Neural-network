# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 20:56:06 2022

@author: Foroogh
"""



# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:17:56 2021

@author: Foroogh
"""
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import silhouette_score

from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import plot_confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import cluster, datasets

"""
 ---- Loading data ----
"""
file = pd.read_csv('SignalWithNoise.csv' ,skiprows=0)
#aaa=file[0].to_csv()
data = file.to_numpy()
divData=(data).reshape(1890,10) 
#Data index 0 signal with noise , index 1 signal without noise 


"""
 ---- Define features ----
"""

meanDD=np.ones(1890)
maxDD=np.ones(1890)
medianDD=np.ones(1890)
rmsDD=np.ones(1890)
kurDD=np.ones(1890)
skewDD=np.ones(1890)
indexDD=np.ones(1890)
for i in range(divData.shape[0]):
    indexDD[i]=i;
    meanDD[i]=divData[i].mean()
    maxDD[i]=max(divData[i])
    medianDD[i]=np.median(divData[i])
    rmsDD[i]=np.sqrt(np.mean(divData[i]**2))
    kurDD[i]=kurtosis(divData[i], fisher=False)
    skewDD[i]=skew(divData[i])
    
"""
 ---- Merge features ----
"""

outputC=np.zeros((1890,2));

mergeFtrsC=np.zeros((1890,7)) #empty matrix
mergeFtrsR=np.vstack([indexDD, maxDD,rmsDD,meanDD,kurDD,skewDD,medianDD]) 
for i in range(len(mergeFtrsR)):
   for j in range(len(mergeFtrsR[0])):
       mergeFtrsC[j][i] = mergeFtrsR[i][j]

"""
 ---- Shuffle ----

"""
per_list = np.random.permutation(len(mergeFtrsC))
sh_inputs = []
for i in range(len(per_list)):
    temp = per_list[i]
    tmp_inputs = mergeFtrsC[temp]
    sh_inputs.append(tmp_inputs)
sh_inputs = np.array(sh_inputs)  # شافل ورودی

"""
 ---- Train Test ----
"""

split_border = int(0.5 * len(sh_inputs)) 

x_train=sh_inputs[0:split_border, :] 

x_val= sh_inputs[split_border: , :]

x_train=x_train[x_train[:, 0].argsort()] 
x_val=x_val[x_val[:, 0].argsort()]
#-------------------------------------------------------------


model = KMeans(n_clusters=4 ).fit(x_train[:,1:])

def sorted_cluster(x, model=None):
    if model == None:
        model = KMeans()
    model = sorted_cluster_centers_(model, x)
    model = sorted_labels_(model, x)
    return model

def sorted_cluster_centers_(model, x):
    model.fit(x)
    magnitude = []
    for center in model.cluster_centers_:
        magnitude.append(np.sqrt(center.dot(center)))
    idx_argsort = np.argsort(magnitude)
    model.cluster_centers_ = model.cluster_centers_[idx_argsort]
    return model
def sorted_labels_(sorted_model, x):
    sorted_model.labels_ = sorted_model.predict(x)
    return sorted_model
model = sorted_cluster(x_train[:,1:], model)
y = model.predict(x_val[:,1:])
y1=y+1



#--------------------  Elbow Method  ---------------------------------------
distortions = []
K = range(2,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(x_train[:,1:])
    kmeanModel = sorted_cluster(x_train[:,1:], kmeanModel)
    predd=kmeanModel.predict(x_val[:,1:])
    score1 =silhouette_score(x_val[:,1:], predd)
    print('Score: %.3f' % score1)
    distortions.append(kmeanModel.inertia_)
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()