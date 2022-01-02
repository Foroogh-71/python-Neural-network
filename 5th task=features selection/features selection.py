# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 08:37:29 2021

@author: Foroogh
"""
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import plot_confusion_matrix
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
"""
 ---- Loading data ----
"""
file = pd.read_excel('signal.xlsx' ,skiprows=0)
data = file.to_numpy()
"""
 ---- input output ----
"""
divData=(data[0]).reshape(210,10) 
#Data index 0 signal with noise , index 1 signal without noise 
input=divData
output=np.ones((210))
for i in range(0,100):
    output[i]=1
for i in range(100,180):
    output[i]=2
for i in range(180,210):
    output[i]=3

"""
 ---- Define features ----
"""

meanDD=np.ones(210)
maxDD=np.ones(210)
medianDD=np.ones(210)
rmsDD=np.ones(210)
kurDD=np.ones(210)
skewDD=np.ones(210)
indexDD=np.ones(210)
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

outputC=np.zeros((210,2));

mergeFtrsC=np.zeros((210,7)) #empty matrix
outputR=np.zeros((210,2))    #empty matrix

mergeFtrsR=np.vstack([indexDD, skewDD,meanDD,maxDD,medianDD,rmsDD,kurDD ]) 
outputR=np.vstack([indexDD,output]) 


for i in range(len(mergeFtrsR)):
   # iterate through columns
   for j in range(len(mergeFtrsR[0])):
       mergeFtrsC[j][i] = mergeFtrsR[i][j]
#outputF=np.hstack((mergeFtrsC, output))


for i in range(len(outputR)):
   # iterate through columns
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
"""
 ---- features selection ----
"""
"""
#number 1. Univariate Selection
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(x_train[:, 1:],y_train[:, 1:])
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(x_train[:, 1:])
# summarize selected features
print(features[0:5,:])
"""

"""
pca = PCA(n_components=4)
fit = pca.fit(x_train[:, 1:])
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
fitC=fit.components_
print(fit.components_)
"""
"""
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 4)
fit = rfe.fit(x_train[:, 1:],y_train[:, 1:])
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
"""

model = ExtraTreesClassifier(n_estimators=10)
model.fit(x_train[:, 1:],y_train[:, 1:].ravel())
print(model.feature_importances_)
