# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:57:15 2021

@author: Foroogh
"""

import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

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

split_border = int(0.9 * len(sh_inputs)) 

x_train=sh_inputs[0:split_border, :] 

x_val= sh_inputs[split_border: , :]

x_train=x_train[x_train[:, 0].argsort()] 
x_val=x_val[x_val[:, 0].argsort()]

"""
------ Agglomerative Clustering -------
"""

cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', compute_full_tree=True, distance_threshold=200)

# Cluster the data
cluster.fit_predict(x_train[:,1:])

print(f"Number of clusters = {1+np.amax(cluster.labels_)}")

# Display the clustering, assigning cluster label to every datapoint 
print("Classifying the points into clusters:")
print(cluster.labels_)

# Display the clustering graphically in a plot
plt.scatter(x_train[:,1:2],x_train[:,2:3], c=cluster.labels_, cmap='rainbow')
plt.title(f"SK Learn estimated number of clusters = {1+np.amax(cluster.labels_)}")
plt.show()

print(" ")

#model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(cluster, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()