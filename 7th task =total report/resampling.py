# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:49:20 2022

@author: Foroogh
"""

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
file = pd.read_csv('Simulation.csv' ,skiprows=0)
data = file.to_numpy()
s0=data[:,0]
s1=data[:,1]
s2=data[:,2]
s3=data[:,3]
#s4=data[:,4]

new_series = pd.Series(data[:,0],index=pd.period_range('2018-01-01', freq='Q', periods=20999))
upsampled = new_series.resample('M')
interpolated = upsampled.interpolate(method='linear').to_numpy()

np.savetxt("Signalnumber231.csv", interpolated, delimiter=",")