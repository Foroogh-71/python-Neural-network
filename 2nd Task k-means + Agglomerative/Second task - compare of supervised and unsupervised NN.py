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
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import plot_confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import cluster, datasets

"""
 ---- Loading data ----
"""
file = pd.read_excel('signal.xlsx' ,skiprows=0)
data = file.to_numpy()
"""
 ---- input output ----
"""
xxx=data[0]
divData=(data[0]).reshape(420,5) 
#Data index 0 signal with noise , index 1 signal without noise 
input=divData
output=np.ones((420))
for i in range(0,200):
    output[i]=1
for i in range(200,360):
    output[i]=2
for i in range(360,420):
    output[i]=3

"""
 ---- Define features ----
"""

meanDD=np.ones(420)
maxDD=np.ones(420)
medianDD=np.ones(420)
rmsDD=np.ones(420)
kurDD=np.ones(420)
skewDD=np.ones(420)
indexDD=np.ones(420)
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

outputC=np.zeros((420,2));

mergeFtrsC=np.zeros((420,3)) #empty matrix
mergeFtrsR=np.vstack([indexDD, maxDD,rmsDD]) 
outputR=np.zeros((420,2))    #empty matrix
outputR=np.vstack([indexDD,output]) 
for i in range(len(mergeFtrsR)):
   for j in range(len(mergeFtrsR[0])):
       mergeFtrsC[j][i] = mergeFtrsR[i][j]
for i in range(len(outputR)):
   for j in range(len(outputR[0])):
       outputC[j][i]=outputR[i][j]
"""
 ---- Shuffle ----

"""
per_list = np.random.permutation(len(mergeFtrsC))
sh_inputs = []
sh_outputs = []
for i in range(len(per_list)):
    temp = per_list[i]
    tmp_inputs = mergeFtrsC[temp]
    tmp_outputs = outputC[temp]
    sh_inputs.append(tmp_inputs)
    sh_outputs.append(tmp_outputs)
sh_inputs = np.array(sh_inputs)  # شافل ورودی
sh_outputs = np.array(sh_outputs)  # شافل خروجی

"""
 ---- Train Test ----
"""

split_border = int(0.7 * len(sh_inputs)) 

x_train=sh_inputs[0:split_border, :] 
y_train=sh_outputs[0:split_border , :]

x_val= sh_inputs[split_border: , :]
y_val=sh_outputs[split_border:,:]

x_train=x_train[x_train[:, 0].argsort()] 
y_train=y_train[y_train[:, 0].argsort()]
x_val=x_val[x_val[:, 0].argsort()]
y_val=y_val[y_val[:, 0].argsort()]

"""
 ---- MLP Classifier ----
"""

clf = MLPClassifier(max_iter=1000 ).fit(x_train[:,1:],y_train[:,1:].ravel())
y_pred=clf.predict(x_val[:,1:])
#score supervised learning
s=accuracy_score(y_val[:,1:], y_pred)
print(s)

"""
 ---- K-Means Clustering ----
"""

model = KMeans(n_clusters=3,init='k-means++',random_state=0).fit(x_train[:,1:])

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

#score UNsupervised learning
us1=accuracy_score(y_val[:,1:], y1)
print(us1)
"""
 ---- Agglomerative Clustering ----
"""

CAC = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward').fit(x_val[:,1:])

"""
 ---- Plot ----
"""
f=plt.figure()
ax=f.add_subplot(2,1,1 )
ax.plot(y_val[:, 1],linewidth=1,linestyle='none',marker='o',markersize=9 ,color="green")# orginal
ax.plot(y_pred,linewidth=1,linestyle='none',marker='o',markersize=6,color="orange" , label='predict') #predict
ax.plot(y1,linewidth=1,linestyle='none',marker='o',markersize=2,color="black" , label='predict') #predict
#----------------------------------------------
plot_confusion_matrix(clf,x_val[:,1:],y_val[:,1:],display_labels=['Light Noise', 'Mean Noise', 'High Noise'],include_values=bool)  
plt.show()
#----------------------------------------------
f=plt.figure()
ax=f.add_subplot(2,1,1 )
ax.scatter(x_val[:,1], x_val[:,2], s=50, c=y,cmap='viridis')
#ax.set_xlabel('max')
ax.set_ylabel('RMS')
ax.set_title('K-Means Clustering ');

plt.subplot(212)
plt.scatter(x_val[:,1], x_val[:,2], s=50, c=CAC.labels_)
plt.xlabel('max')
plt.ylabel('RMS')
plt.title("Agglomerative Clustering")