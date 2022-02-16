# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 22:20:41 2022

@author: Foroogh
"""


from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
import numpy as np
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('clear')


plt.close('all')


def color(intnumber):
    if (intnumber == 0):
        color = 'black'
    if (intnumber == 1):
        color = 'r'
    if (intnumber == 2):
        color = 'm'
    if (intnumber == 3):
        color = 'c'
    if (intnumber == 4):
        color = 'b'
    if (intnumber == 5):
        color = 'g'
    if (intnumber == 6):
        color = 'k'
    return color


"""
 ---- Loading data ----
"""
file = pd.read_csv('Gaussian.csv', skiprows=0)
# aaa=file[0].to_csv()
dataa = file.to_numpy()
# divData=(data).reshape(189,100)
# Data index 0 signal with noise , index 1 signal without noise
data = np.zeros((1000, 980))
data = dataa.transpose()
signal1 = data[:, 0]
#data[:, 1] = dataa[1, :]
signal = data[:, 1]

# overlaping
"""
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
numberOfSeg = 50
divData = (data[:, 1]).reshape(50, 20)

"""
 ---- Define features ----
"""
stdDD = np.ones(numberOfSeg)
meanDD = np.ones(numberOfSeg)
maxDD = np.ones(numberOfSeg)
medianDD = np.ones(numberOfSeg)
rmsDD = np.ones(numberOfSeg)
kurDD = np.ones(numberOfSeg)
skewDD = np.ones(numberOfSeg)
indexDD = np.ones(numberOfSeg)
for i in range(divData.shape[0]):
    indexDD[i] = i
    meanDD[i] = divData[i].mean()
    maxDD[i] = max(divData[i])
    medianDD[i] = np.median(divData[i])
    rmsDD[i] = np.sqrt(np.mean(divData[i]**2))
    kurDD[i] = kurtosis(divData[i], fisher=False)
    skewDD[i] = skew(divData[i])
    stdDD[i] = np.std(divData[i])


"""
 #---- Merge features ----
"""

outputC = np.zeros((numberOfSeg, 2))

mergeFtrsC = np.zeros((numberOfSeg, 8))  # empty matrix
mergeFtrsR = np.vstack([indexDD, stdDD, meanDD, rmsDD,
                       kurDD, skewDD, medianDD, maxDD])
for i in range(len(mergeFtrsR)):
    for j in range(len(mergeFtrsR[0])):
        mergeFtrsC[j][i] = mergeFtrsR[i][j]
np.savetxt("ftrsC.csv", mergeFtrsC, delimiter=",")
featuresX = 1
featuresY = 2
string1 = 'STD'
string2 = 'Mean'
string3 = 'RMS'


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

x_train = sh_inputs[0:split_border, :]

x_val = sh_inputs[split_border:, :]

x_train = x_train[x_train[:, 0].argsort()]
x_val = x_val[x_val[:, 0].argsort()]
# ----------------------------------------------------

# -----------------------------------------------------

pca = PCA(n_components=7)
fit = pca.fit(x_train[:, 1:])
# summarize components
a = fit.explained_variance_ratio_ * 100
plt.bar(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], a)
# for index, value in enumerate(a):
#    plt.text(value, index, str(value))
print("Explained Variance: %s" % fit.explained_variance_ratio_)
fitC = fit.components_
# print(fit.components_)

pcaTrain = np.zeros((25, 8))
pcaTrain[:, 0] = x_train[:, 0]
pcaTrain[:, 1:] = fit.fit_transform(x_train[:, 1:])
# -----------------------------------
pcatest = np.zeros((25, 8))
pcatest[:, 0] = x_val[:, 0]
pcatest[:, 1:] = fit.fit_transform(x_val[:, 1:])
# ------------------------------------------------------
matrix = np.zeros((25, 21))
matrixtrain = np.zeros((25, 21))

#
matrix[:, 0] = pcatest[:, 0]
for i in range(50):
    for j in range(25):
        if(i == (x_val[j, 0])):
            matrix[j, 1:] = divData[i]


matrixtrain[:, 0] = pcaTrain[:, 0]
for i in range(50):
    for j in range(25):
        if(i == (x_train[j, 0])):
            matrixtrain[j, 1:] = divData[i]

figpca, axspca = plt.subplots(4, 2)
figpca.suptitle('using PCA')
t = np.linspace(0, 1000, 1000)
axspca[0][0].plot(t, signal)
axspca[1][0].scatter(pcatest[:, 0]*50, pcatest[:, 1])
axspca[2][0].scatter(pcatest[:, 0]*50, pcatest[:, 2])
axspca[3][0].scatter(pcatest[:, 0]*50, pcatest[:, 3])
axspca[0][1].scatter(pcatest[:, 0]*50, pcatest[:, 4])
axspca[1][1].scatter(pcatest[:, 0]*50, pcatest[:, 5])
axspca[2][1].scatter(pcatest[:, 0]*50, pcatest[:, 6])
axspca[3][1].scatter(pcatest[:, 0]*50, pcatest[:, 7])

axspca[0][0].set_ylabel('signal')
axspca[1][0].set_ylabel('pc1')
axspca[2][0].set_ylabel('pc2')
axspca[3][0].set_ylabel('pc3')
axspca[0][1].set_ylabel('pc4')
axspca[1][1].set_ylabel('pc5')
axspca[2][1].set_ylabel('pc6')
axspca[3][1].set_ylabel('pc7')

axspca[3][0].set_xlabel('datapoints')
axspca[3][1].set_xlabel('datapoints')

"""
#--------------------  Elbow Method  ---------------------------------------
distortions = []
K = range(2,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(pcaTrain[:,1:])
   # kmeanModel = sorted_cluster(pcaTrain[:,1:], kmeanModel)
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
# ------ dendrogram -------
"""

cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', compute_full_tree=True, distance_threshold=3)

# Cluster the data
cluster.fit_predict(pcaTrain[:,1:])


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
"""
xs = pcatest[:, 1]
xst = pcaTrain[:, 1]


k = 4
# -------------------------------------------------------------

# spectral clustering
# define the model
model = SpectralClustering(n_clusters=k)
# fit model and predict clusters
y1 = model.fit_predict(pcaTrain[:, 1:2])
yhat1 = model.fit_predict(pcatest[:, 1:2])


#score UNsupervised learning
# Calculate Silhoutte Score
# score =silhouette_score(x_val[:,1:], yhat1)
# print('spectral clustering for k:',k,' Score=%.3f' % (score*100), '%' )

#----------------------------------------------------
# birch clustering
# define the model
model = Birch(threshold=0.01, n_clusters=k)
# fit the model
y2=model.fit_predict(pcaTrain[:,1:2])
yhat2 = model.predict(pcatest[:, 1:2])


# score =silhouette_score(x_val[:,1:], yhat2)
# print('birch clustering for k:',k,' Score=%.3f' % (score*100), '%' )


#----------------------------------------------------
# agglomerative clustering

# define the model
model = AgglomerativeClustering(n_clusters=k)
# fit model and predict clusters
yhat3 = model.fit_predict(pcaTrain[:, 1:2])
yhat3 = model.fit_predict(pcatest[:, 1:2])


# score =silhouette_score(x_val[:,1:], yhat3)
# print('agglomerative clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#----------------------------------------------------
# affinity propagation clustering
# define the model
model = AffinityPropagation(damping=0.7,random_state=0)
y4=model.fit_predict(pcaTrain[:, 1:2])
yhat4 = model.predict(pcatest[:, 1:2])


# score =silhouette_score(x_val[:,1:], yhat4)
# print('affinity propagation clustering Score=%.3f' % (score*100), '%' )
#---------------------------------------------------------------

# dbscan clustering
model = DBSCAN(eps=0.010, min_samples=9)
y5=(model.fit_predict(pcaTrain[:, 1:2]))
yhat5=(model.fit_predict(pcatest[:, 1:2]))


#score =silhouette_score(x_val[:,1:], yhat5)
#print('DB scan clustering Score=%.3f' % (score*100), '%' )


#---------------------------------------------------------------
# k-means clustering
model = KMeans(n_clusters=k)

y6 = model.fit_predict(pcaTrain[:, 1:2])
# assign a cluster to each example
yhat6 = model.predict(pcatest[:, 1:2])

# score =silhouette_score(x_val[:,1:], yhat6)
# print('k-means clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#---------------------------------------------------------------

# mini-batch k-means clustering
model = MiniBatchKMeans(n_clusters=k)
y7=model.fit_predict(pcaTrain[:, 1:2])
yhat7 = model.predict(pcatest[:, 1:2])

# score =silhouette_score(x_val[:,1:], yhat7)
# print('mini-batch k-means clustering for k:',k,' Score=%.3f' % (score*100), '%' )
#---------------------------------------------------------------
# optics clustering
# model = OPTICS(eps=0.8, min_samples=10)
# y8=model.fit_predict(pcaTrain[:, 1:2])
# yhat8 = model.fit_predict(x_val[:,1:])


# score =silhouette_score(x_val[:,1:], yhat8)
# print('optics clustering Score=%.3f' % (score*100), '%' )

# ----------------------------------------------------------------------------

#mean shift clustering
model = MeanShift()
y9=model.fit_predict(pcaTrain[:, 1:2])
yhat9 = model.predict(pcatest[:, 1:2])


# score =silhouette_score(x_val[:,1:], yhat9)
# print('mean shift clustering Score=%.3f' % (score*100), '%' )

# ----------------------------------------------------------------------------

# gaussian mixture clustering
model = GaussianMixture(n_components=k)
# fit the model
trainresult = model.fit_predict(pcaTrain[:, 1:2])
# assign a cluster to each example
yhat10 = model.predict(pcatest[:, 1:2])

score = silhouette_score(pcatest[:, 1:2], yhat10)
print('gaussian mixture clustering Score=%.3f' % (score*100), '%')

# ----------------------------------------------------------------------------

df = pd.DataFrame(yhat1)
df.insert(loc=0, column='', value=pcatest[:, 0])
yhat1 = df.to_numpy()

df = pd.DataFrame(yhat2)
df.insert(loc=0, column='', value=pcatest[:, 0])
yhat2 = df.to_numpy()

df = pd.DataFrame(yhat3)
df.insert(loc=0, column='', value=pcatest[:, 0])
yhat3 = df.to_numpy()

df = pd.DataFrame(yhat4)
df.insert(loc=0, column='', value=pcatest[:, 0])
yhat4 = df.to_numpy()

df = pd.DataFrame(yhat5)
df.insert(loc=0, column='', value=pcatest[:, 0])
yhat5 = df.to_numpy()

df = pd.DataFrame(yhat6)
df.insert(loc=0, column='', value=pcatest[:, 0])
yhat6 = df.to_numpy()

df = pd.DataFrame(yhat7)
df.insert(loc=0, column='', value=pcatest[:, 0])
yhat7 = df.to_numpy()

df = pd.DataFrame(yhat9)
df.insert(loc=0, column='', value=pcatest[:, 0])
yhat9 = df.to_numpy()

df = pd.DataFrame(yhat10)
df.insert(loc=0, column='', value=pcatest[:, 0])
yhat10 = df.to_numpy()

#------------------------------------------------------

df = pd.DataFrame(trainresult)
df.insert(loc=0, column='', value=pcaTrain[:, 0])
trainresult = df.to_numpy()

df = pd.DataFrame(y1)
df.insert(loc=0, column='', value=pcaTrain[:, 0])
y1 = df.to_numpy()

df = pd.DataFrame(y2)
df.insert(loc=0, column='', value=pcaTrain[:, 0])
y2 = df.to_numpy()

# df = pd.DataFrame(y3)
# df.insert(loc=0, column='', value=pcaTrain[:, 0])
# y3 = df.to_numpy()

df = pd.DataFrame(y4)
df.insert(loc=0, column='', value=pcaTrain[:, 0])
y4 = df.to_numpy()

df = pd.DataFrame(y5)
df.insert(loc=0, column='', value=pcaTrain[:, 0])
y5 = df.to_numpy()

df = pd.DataFrame(y6)
df.insert(loc=0, column='', value=pcaTrain[:, 0])
y6 = df.to_numpy()

df = pd.DataFrame(y7)
df.insert(loc=0, column='', value=pcaTrain[:, 0])
y7 = df.to_numpy()

# df = pd.DataFrame(y8)
# df.insert(loc=0, column='', value=pcaTrain[:, 0])
# y8 = df.to_numpy()

df = pd.DataFrame(y9)
df.insert(loc=0, column='', value=pcaTrain[:, 0])
y9 = df.to_numpy()

#--------------------------------------------------------------------------------
fig6, axs6 = plt.subplots(4, 2)
fig6.suptitle('using PCA')

t = np.linspace(0, 1000, 1000)
axs6[0][0].plot(t, signal)
axs6[0][0].set_title('title= signal without color')
axs6[0][0].set_ylabel('amount')
axs6[0][0].set_xlabel('number of datapoint')


for i in range(0, 25):
    axs6[1][0].plot(range((int(matrix[i, 0]))*20, int(matrix[i, 0]+1)
                 * 20), matrix[i, 1:], color(yhat10[i, 1]))
for i in range(0, 25):
    axs6[1][0].plot(range((int(matrixtrain[i, 0]))*20, int(matrixtrain[i, 0]+1)
                 * 20), matrixtrain[i, 1:], color(trainresult[i, 1]))
axs6[1][0].set_ylabel('gaussian clustering')
#----------------------------------
for i in range(0, 25):
    axs6[2][0].plot(range((int(matrix[i, 0]))*20, int(matrix[i, 0]+1)
                 * 20), matrix[i, 1:], color(yhat1[i, 1]))
for i in range(0, 25):
    axs6[2][0].plot(range((int(matrixtrain[i, 0]))*20, int(matrixtrain[i, 0]+1)
                 * 20), matrixtrain[i, 1:], color(y1[i, 1]))
axs6[2][0].set_ylabel('spectral clustering')
#-----------------------------------
for i in range(0, 25):
    axs6[3][0].plot(range((int(matrix[i, 0]))*20, int(matrix[i, 0]+1)
                 * 20), matrix[i, 1:], color(yhat2[i, 1]))
for i in range(0, 25):
    axs6[3][0].plot(range((int(matrixtrain[i, 0]))*20, int(matrixtrain[i, 0]+1)
                 * 20), matrixtrain[i, 1:], color(y2[i, 1]))
axs6[3][0].set_ylabel('birch clustering')
#----------------------------------
for i in range(0, 25):
    axs6[0][1].plot(range((int(matrix[i, 0]))*20, int(matrix[i, 0]+1)
                 * 20), matrix[i, 1:], color(yhat4[i, 1]))
for i in range(0, 25):
    axs6[0][1].plot(range((int(matrixtrain[i, 0]))*20, int(matrixtrain[i, 0]+1)
                 * 20), matrixtrain[i, 1:], color(y4[i, 1]))
axs6[0][1].set_ylabel('affinity propagation')
#-----------------------------------
for i in range(0, 25):
    axs6[1][1].plot(range((int(matrix[i, 0]))*20, int(matrix[i, 0]+1)
                 * 20), matrix[i, 1:], color(yhat6[i, 1]))
for i in range(0, 25):
    axs6[1][1].plot(range((int(matrixtrain[i, 0]))*20, int(matrixtrain[i, 0]+1)
                 * 20), matrixtrain[i, 1:], color(y6[i, 1]))
axs6[1][1].set_ylabel('k-means')
#----------------------------------
for i in range(0, 25):
    axs6[2][1].plot(range((int(matrix[i, 0]))*20, int(matrix[i, 0]+1)
                 * 20), matrix[i, 1:], color(yhat7[i, 1]))
for i in range(0, 25):
    axs6[2][1].plot(range((int(matrixtrain[i, 0]))*20, int(matrixtrain[i, 0]+1)
                 * 20), matrixtrain[i, 1:], color(y7[i, 1]))
axs6[2][1].set_ylabel('mini-batch k-means')
#----------------------------------
for i in range(0, 25):
    axs6[3][1].plot(range((int(matrix[i, 0]))*20, int(matrix[i, 0]+1)
                 * 20), matrix[i, 1:], color(yhat9[i, 1]))
for i in range(0, 25):
    axs6[3][1].plot(range((int(matrixtrain[i, 0]))*20, int(matrixtrain[i, 0]+1)
                 * 20), matrixtrain[i, 1:], color(y9[i, 1]))
axs6[3][1].set_ylabel('mean shift ')
#-----------------------------------

#------------plot-----------------------
plot1 = pyplot. figure('orginal')
plt.title('title= signal')
plt.ylabel('amount')
plt.xlabel('datapoints')

t1 = np.linspace(0, 350, 350)
t2 = np.linspace(350,700,350) 
t3 = np.linspace(700,1000,300) 
  
plt.plot(t1,data[0:350, 1] ,'b' )
plt.plot(t2,data[350:700, 1] ,'black' )
plt.plot(t3,data[700:1000, 1] ,'r' )
#----------End plot----------------------


firstIndex=np.zeros((980,4))
for N in range (0, 8):
    
    plot1 = pyplot. figure('signal')
    plt.title('title= signal')
    plt.ylabel('amount')
    plt.xlabel('datapoints')
    t = np.linspace(0, 1000, 1000)
    plt.plot(t,data[:, N] )
    
    numberOfSeg = 50
    divData = (data[:, N]).reshape(50, 20)

    stdDD = np.ones(numberOfSeg)
    meanDD = np.ones(numberOfSeg)
    maxDD = np.ones(numberOfSeg)
    medianDD = np.ones(numberOfSeg)
    rmsDD = np.ones(numberOfSeg)
    kurDD = np.ones(numberOfSeg)
    skewDD = np.ones(numberOfSeg)
    indexDD = np.ones(numberOfSeg)
    for i in range(divData.shape[0]):
        indexDD[i] = i
        meanDD[i] = divData[i].mean()
        maxDD[i] = max(divData[i])
        medianDD[i] = np.median(divData[i])
        rmsDD[i] = np.sqrt(np.mean(divData[i]**2))
        kurDD[i] = kurtosis(divData[i], fisher=False)
        skewDD[i] = skew(divData[i])
        stdDD[i] = np.std(divData[i])

    mergeFtrsC = np.zeros((numberOfSeg, 8))  # empty matrix
    mergeFtrsR = np.vstack([indexDD, stdDD, meanDD, rmsDD,
                           kurDD, skewDD, medianDD, maxDD])
    for i in range(len(mergeFtrsR)):
        for j in range(len(mergeFtrsR[0])):
            mergeFtrsC[j][i] = mergeFtrsR[i][j]
    

    per_list = np.random.permutation(len(mergeFtrsC))
    sh_inputs = []
    for i in range(len(per_list)):
        temp = per_list[i]
        tmp_inputs = mergeFtrsC[temp]
        sh_inputs.append(tmp_inputs)
    sh_inputs = np.array(sh_inputs)  # شافل ورودی
    split_border = int(0.5 * len(sh_inputs)) 
    x_train=sh_inputs[0:split_border, :] 
    x_val= sh_inputs[split_border: , :]
    x_train=x_train[x_train[:, 0].argsort()] 
    x_val=x_val[x_val[:, 0].argsort()]

    pca = PCA(n_components=7)
    fit = pca.fit(x_train[:, 1:])
    pcaTrain[:, 1:]=fit.fit_transform(x_train[:, 1:])
    pcatest[:, 0]=x_val[:, 0]
    pcatest[:, 1:]=fit.fit_transform(x_val[:, 1:])
   # model = GaussianMixture(n_components=k)
    model = KMeans(n_clusters=k)
    #model = sorted_cluster(pcaTrain[:,1:], model)
    model=model.fit(pcaTrain[:,1:])
    yhat10 = model.predict(pcatest[:,1:])
    df = pd.DataFrame(yhat10)
    df.insert(loc=0, column='', value=pcatest[:,0])
    yhat10 = df.to_numpy()
    new=yhat10[yhat10[:, 0].argsort()] 
    inx=np.zeros(4)
    inx0=(np.asarray(np.where(new[:,1] == 0))).transpose() 
    inx1=(np.asarray(np.where(new[:,1] == 1))).transpose() 
    inx2=(np.asarray(np.where(new[:,1] == 2))).transpose() 
    inx3=(np.asarray(np.where(new[:,1] == 3))).transpose()
    #inx4=(np.asarray(np.where(new[:,1] == 3))).transpose()
    if (len(inx0)>0):

      inx[0]=inx0[0]
    if (len(inx1)>0):

      inx[1]=inx1[0]
    if (len(inx2)>0):

      inx[2]=inx2[0]
    if (len(inx3)>0):

      inx[3]=inx3[0]
    inx = inx[inx[:].argsort()]

    if (len(inx0)>0):
        firstIndex[N,0]=inx[0]
    if (len(inx1)>0):
        firstIndex[N,1]=inx[1]
    if (len(inx2)>0):
        firstIndex[N,2]=inx[2]
    if (len(inx3)>0):
        firstIndex[N,3]=inx[3]
        
    #firstIndex[j,4]=inx4[0]

fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(firstIndex[:,1:],labels =["Class 1",  "Class 2", "Class 1"])
plt.ylabel("Number of segments")
#plt.xlabel("Classes")
# show plot
plt.show()