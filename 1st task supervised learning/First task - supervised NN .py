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

mergeFtrsR=np.vstack([indexDD, meanDD,maxDD,medianDD,rmsDD,kurDD,skewDD ]) 
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
 ---- MLP Classifier ----
"""

clf = MLPClassifier(max_iter=1000 ).fit(x_train[:, 1:],y_train[:, 1:].ravel())
y_pred=clf.predict(x_val[:,1:])


outputC=np.zeros((63,3)); #saving real index for Y_pred
for i in range((63)):
   outputC[i][2]=y_pred[i]
   for j in range(2):
       outputC[i][j] = y_val[i][j]
       
#score
score=clf.score(x_val[:,1:],outputC[:,1:2]);
print(score)
#

FinalOutPutplt=outputC[outputC[:, 0].argsort()] # sorting by real index

"""
 ---- Plot ----
"""
f=plt.figure()
ax=f.add_subplot(4,1,1 )
ax.plot(FinalOutPutplt[:, 1:2],linewidth=1,linestyle='none',marker='o',markersize=13 ,color="green")# orginal
ax.plot(FinalOutPutplt[:, 2:],linewidth=1,linestyle='none',marker='o',markersize=7,color="orange" , label='predict') #predict

plt.subplot(412 ,facecolor='lightgreen')
plt.plot(data[0], label= 'signal' , color='black')

plt.subplot(413)
plt.plot(data[1], label= 'signal' ,color='blue')

plt.subplot(414,facecolor='lightyellow')
plt.plot(data[2], label= 'signal' ,color='darkred')

plot_confusion_matrix(clf,x_val[:,1:],outputC[:, 1:2],display_labels=['Light Noise', 'Mean Noise', 'High Noise'],include_values=bool)  
plt.show()