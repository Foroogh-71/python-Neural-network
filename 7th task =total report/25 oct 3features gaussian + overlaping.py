# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:14:10 2022

@author: Foroogh
"""
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('clear')


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
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA

plt.close('all')

def color(intnumber):
    if (intnumber==0):
        color='b';
    if (intnumber==1):
        color='r'
    if (intnumber==2):
        color='g'
    if (intnumber==3):
        color='y'
    if (intnumber==4):
        color='c'
    if (intnumber==5):
        color='m'
    if (intnumber==6):
        color='k'
    return color


"""
 ---- Loading data ----
"""
file = pd.read_csv('Signalnumber231.csv' ,skiprows=0)
#aaa=file[0].to_csv()
data = file.to_numpy()
#divData=(data).reshape(189,100) 
#Data index 0 signal with noise , index 1 signal without noise 


# overlaping 
datapoint=0
first=50
end=50
windowsize=100 # size of window
numberOfSeg=int((62997/(first))-(end/first))
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

mergeFtrsC=np.zeros((numberOfSeg,4)) #empty matrix
mergeFtrsR=np.vstack([indexDD,stdDD,meanDD,rmsDD]) 
for i in range(len(mergeFtrsR)):
   for j in range(len(mergeFtrsR[0])):
       mergeFtrsC[j][i] = mergeFtrsR[i][j]
np.savetxt("ftrsC.csv", mergeFtrsC, delimiter=",")
featuresX=1
featuresY=2
string1='STD'
string2='Mean'
string3='RMS'


"""
 #---- Shuffle ----

"""
per_list = np.random.permutation(len(mergeFtrsC))
sh_inputs = []
for i in range(len(per_list)):
    temp = per_list[i]
    tmp_inputs = mergeFtrsC[temp]
    sh_inputs.append(tmp_inputs)
sh_inputs = np.array(sh_inputs)  # ???????? ??????????

"""
 #---- Train Test ----
"""

split_border = int(0.5 * len(sh_inputs)) 

x_train=sh_inputs[0:split_border, :] 

x_val= sh_inputs[split_border: , :]

x_train=x_train[x_train[:, 0].argsort()] 
x_val=x_val[x_val[:, 0].argsort()]
#----------------------------------------------------
matrix=np.zeros((629,104));

#
matrix[:,:4]=x_val
for i in range(1258):
    for j in range(629):
      if(i== (x_val[j,0])):
         matrix[j,4:]=divData[i]
#-----------------------------------------------------

pca = PCA(n_components=3)
fit = pca.fit(x_train[:, 1:])
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
fitC=fit.components_
#print(fit.components_)
yyyy=fit.fit_transform(x_train[:, 1:])

#--------------------  Elbow Method  ---------------------------------------
distortions = []
K = range(2,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(x_train[:,1:])
   # kmeanModel = sorted_cluster(x_train[:,1:], kmeanModel)
    predd=kmeanModel.predict(x_val[:,1:])
    score1 =silhouette_score(x_val[:,1:], predd)
    print('Score: %.3f' % score1)
    distortions.append(kmeanModel.inertia_)
plotelbow = pyplot. figure('elbow')
plt.figure('elbow',figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
#----------------------------------------------------------------
"""
------ dendrogram -------
"""

cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', compute_full_tree=True, distance_threshold=3)

# Cluster the data
cluster.fit_predict(x_train[:,1:])


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
plot1 = pyplot. figure('dendrogram')

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(cluster, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#----------------------------------------------------------------
xs = x_val[:, 1]

ys = x_val[:, 2]

zs = x_val[:, 3]



k=4;
#-------------------------------------------------------------
# spectral clustering
# define the model
model = SpectralClustering(n_clusters=k)
# fit model and predict clusters
yhat1 = model.fit_predict(x_val[:,1:])
plot1 = pyplot. figure('spectral clustering')


ax = plt.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat1)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('spectral clustering');
plt.show()


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
plot2 = pyplot. figure('birch clustering')

ax = pyplot.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat2)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('birch clustering');
plt.show()


score =silhouette_score(x_val[:,1:], yhat2)
print('birch clustering for k:',k,' Score=%.3f' % (score*100), '%' )


#----------------------------------------------------
# agglomerative clustering

# define the model
model = AgglomerativeClustering(n_clusters=k)
# fit model and predict clusters
yhat3 = model.fit_predict(x_val[:,1:])

plot3 = pyplot. figure('agglomerative clustering')
ax = pyplot.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat3)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('agglomerative clustering');
plt.show()

score =silhouette_score(x_val[:,1:], yhat3)
print('agglomerative clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#----------------------------------------------------
# affinity propagation clustering
# define the model
model = AffinityPropagation(damping=0.7,random_state=0)
model.fit(x_train[:,1:])
yhat4 = model.predict(x_val[:,1:])

plot4 = pyplot. figure('affinity propagation clustering')
ax = pyplot.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat4)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('affinity propagation clustering');
plt.show()

score =silhouette_score(x_val[:,1:], yhat4)
print('affinity propagation clustering Score=%.3f' % (score*100), '%' )
#---------------------------------------------------------------

# dbscan clustering
model = DBSCAN(eps=0.010, min_samples=9)
model.fit(x_train[:,1:])
yhat5=model.fit_predict(x_val[:,1:])

plot5 = pyplot. figure('dbscan clustering')
ax = pyplot.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat5)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('dbscan clustering');
plt.show()

score =silhouette_score(x_val[:,1:], yhat5)
print('DB scan clustering Score=%.3f' % (score*100), '%' )

#---------------------------------------------------------------
# k-means clustering
model = KMeans(n_clusters=k)
model.fit(x_train[:,1:])
yhat6 = model.predict(x_val[:,1:])

plot6 = pyplot. figure('k-means clustering')
ax = pyplot.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat6)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('k-means clustering');
plt.show()

score =silhouette_score(x_val[:,1:], yhat6)
print('k-means clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#---------------------------------------------------------------
# mini-batch k-means clustering
model = MiniBatchKMeans(n_clusters=k)
model.fit(x_train[:,1:])
yhat7 = model.predict(x_val[:,1:])

plot7 = pyplot. figure('mini-batch k-means clustering')
ax = pyplot.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat7)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('mini-batch k-means clustering');
plt.show()

score =silhouette_score(x_val[:,1:], yhat7)
print('mini-batch k-means clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#---------------------------------------------------------------
# optics clustering
model = OPTICS(eps=0.8, min_samples=10)
model.fit(x_train[:,1:])
yhat8 = model.fit_predict(x_val[:,1:])

plot8 = pyplot. figure('optics clustering')
ax = pyplot.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat8)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('optics clustering');
plt.show()

score =silhouette_score(x_val[:,1:], yhat8)
print('optics clustering Score=%.3f' % (score*100), '%' )
#----------------------------------------------------------------------------
#mean shift clustering
model = MeanShift()
model.fit(x_train[:,1:])
yhat9 = model.fit_predict(x_val[:,1:])

plot9 = pyplot. figure('mean shift clustering')
ax = pyplot.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat9)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('mean shift clustering');
plt.show()

score =silhouette_score(x_val[:,1:], yhat9)
print('mean shift clustering Score=%.3f' % (score*100), '%' )
#----------------------------------------------------------------------------
# gaussian mixture clustering
model = GaussianMixture(n_components=4)
# fit the model
model.fit(x_train[:,1:])
# assign a cluster to each example
yhat10 = model.predict(x_val[:,1:])
plot10 = pyplot. figure('gaussian mixture clustering')
ax = pyplot.axes(projection='3d')
ax.scatter3D(xs,ys,zs, c=yhat10)
ax.set_xlabel(string1)
ax.set_ylabel(string2)
ax.set_zlabel(string3)
ax.set_title('gaussian mixture clustering');
plt.show()
score =silhouette_score(x_val[:,1:], yhat10)
print('gaussian mixture clustering Score=%.3f' % (score*100), '%' )
#----------------------------------------------------------------------------

Feature=0 #index
temp=ys
df = pd.DataFrame(yhat1)
df.insert(loc=0, column='', value=x_val[:,Feature])
yhat1 = df.to_numpy()

df = pd.DataFrame(yhat2)
df.insert(loc=0, column='', value=x_val[:,Feature])
yhat2 = df.to_numpy()

df = pd.DataFrame(yhat3)
df.insert(loc=0, column='', value=x_val[:,Feature])
yhat3 = df.to_numpy()

df = pd.DataFrame(yhat4)
df.insert(loc=0, column='', value=x_val[:,Feature])
yhat4 = df.to_numpy()

df = pd.DataFrame(yhat5)
df.insert(loc=0, column='', value=x_val[:,Feature])
yhat5 = df.to_numpy()

df = pd.DataFrame(yhat6)
df.insert(loc=0, column='', value=x_val[:,Feature])
yhat6 = df.to_numpy()

df = pd.DataFrame(yhat7)
df.insert(loc=0, column='', value=x_val[:,Feature])
yhat7 = df.to_numpy()

df = pd.DataFrame(yhat9)
df.insert(loc=0, column='', value=x_val[:,Feature])
yhat9 = df.to_numpy()

df = pd.DataFrame(yhat10)
df.insert(loc=0, column='', value=x_val[:,Feature])
yhat10 = df.to_numpy()

fig, axs = plt.subplots(3,3)
fig.suptitle('STD')

fig1, axs1 = plt.subplots(3,3)
fig1.suptitle('Mean')

fig2, axs2 = plt.subplots(3,3)
fig2.suptitle('RMS')

axs[0,0].scatter(yhat1[:,0]*50,xs ,c=yhat1[:,1])
axs[0,0].set_title('spectral clustering ')

axs[0,1].scatter(yhat2[:,0]*50,xs ,c=yhat2[:,1])
axs[0,1].set_title('birch clustering ')

axs[0,2].scatter(yhat3[:,0]*50,xs ,c=yhat3[:,1])
axs[0,2].set_title('agglomerative clustering ')

axs[1,0].scatter(yhat4[:,0]*50,xs ,c=yhat4[:,1])
axs[1,0].set_title('affinity clustering ')

axs[1,1].scatter(yhat5[:,0]*50,xs ,c=yhat5[:,1])
axs[1,1].set_title('DB scan clustering ')

axs[1,2].scatter(yhat6[:,0]*50,xs ,c=yhat6[:,1])
axs[1,2].set_title('k-means clustering ')

axs[2,0].scatter(yhat7[:,0]*50,xs ,c=yhat7[:,1])
axs[2,0].set_title('mini-batch k-means clustering ')

axs[2,1].scatter(yhat9[:,0]*50,xs ,c=yhat9[:,1])
axs[2,1].set_title('mean shift clustering ')

axs[2,2].scatter(yhat10[:,0]*50,xs ,c=yhat10[:,1])
axs[2,2].set_title('gaussian mixture clustering ')


axs1[0,0].scatter(yhat1[:,0]*50,ys ,c=yhat1[:,1])
axs1[0,0].set_title('spectral clustering ')

axs1[0,1].scatter(yhat2[:,0]*50,ys ,c=yhat2[:,1])
axs1[0,1].set_title('birch clustering ')

axs1[0,2].scatter(yhat3[:,0]*50,ys ,c=yhat3[:,1])
axs1[0,2].set_title('agglomerative clustering ')

axs1[1,0].scatter(yhat4[:,0]*50,ys ,c=yhat4[:,1])
axs1[1,0].set_title('affinity clustering ')

axs1[1,1].scatter(yhat5[:,0]*50,ys ,c=yhat5[:,1])
axs1[1,1].set_title('DB scan clustering ')

axs1[1,2].scatter(yhat6[:,0]*50,ys ,c=yhat6[:,1])
axs1[1,2].set_title('k-means clustering ')

axs1[2,0].scatter(yhat7[:,0]*50,ys ,c=yhat7[:,1])
axs1[2,0].set_title('mini-batch k-means clustering ')

axs1[2,1].scatter(yhat9[:,0]*50,ys ,c=yhat9[:,1])
axs1[2,1].set_title('mean shift clustering ')

axs1[2,2].scatter(yhat10[:,0]*50,ys ,c=yhat10[:,1])
axs1[2,2].set_title('gaussian mixture clustering ')

axs2[0,0].scatter(yhat1[:,0]*50,zs ,c=yhat1[:,1])
axs2[0,0].set_title('spectral clustering ')

axs2[0,1].scatter(yhat2[:,0]*50,zs ,c=yhat2[:,1])
axs2[0,1].set_title('birch clustering ')

axs2[0,2].scatter(yhat3[:,0]*50,zs ,c=yhat3[:,1])
axs2[0,2].set_title('agglomerative clustering ')

axs2[1,0].scatter(yhat4[:,0]*50,zs ,c=yhat4[:,1])
axs2[1,0].set_title('affinity clustering ')

axs2[1,1].scatter(yhat5[:,0]*50,zs ,c=yhat5[:,1])
axs2[1,1].set_title('DB scan clustering ')

axs2[1,2].scatter(yhat6[:,0]*50,zs ,c=yhat6[:,1])
axs2[1,2].set_title('k-means clustering ')

axs2[2,0].scatter(yhat7[:,0]*50,zs ,c=yhat7[:,1])
axs2[2,0].set_title('mini-batch k-means clustering ')

axs2[2,1].scatter(yhat9[:,0]*50,zs ,c=yhat9[:,1])
axs2[2,1].set_title('mean shift clustering ')

axs2[2,2].scatter(yhat10[:,0]*50,zs ,c=yhat10[:,1])
axs2[2,2].set_title('gaussian mixture clustering ')


fig6, axs6 = plt.subplots(3,1)
fig6.suptitle('using RMS,STD ,Mean ')
axs6[0].scatter(x_val[:, 0]*50 ,zs,c=yhat10[:,1])
axs6[0].set_title('title= for better vision just I plot RMS segment')
axs6[0].set_ylabel('normalized amount')

for i in range(0,629):
    #aaaaaa=range((1)*20+1,2*20)
    #axs6[1].plot(range((int(matrix[i,0])-1)*20+1,int(matrix[i,0])*20),matrix[i,5:],color(yhat7[i,1]))
    axs6[1].plot(range((int(matrix[i,0]))*50,int(matrix[i,0]+1)*50),matrix[i,54:],color(yhat10[i,1]))

    #axs6[1].plot(range(((i-1)*20+1,i*20)),matrix[i,5:],color(yhat7[i,1]))
axs6[1].set_title('title= signal ')
axs6[1].set_ylabel('amount')


t = np.linspace(0, 62996,62996)
axs6[2].plot(t,data)
axs6[2].set_title('title= signal without color')
axs6[2].set_ylabel('amount')
axs6[2].set_xlabel('number of datapoint')