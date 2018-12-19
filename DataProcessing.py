#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:36:45 2018

@author: omerorhan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing dataset
dataset = pd.read_csv('Data.csv')

#missing data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy ='mean', axis =0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,0] =labelencoder_X.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0] )
x =onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y =labelencoder_y.fit_transform(y)

#splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train , y_train =train_test_split(x,y,test_size=0.2, random_state=0)

'''
feature scalling
make sure features are on a similar scale
get every feature into approximately a -1<=xi <=1 range
-100<=x3 <=100   that's wrong. wide range
-2 <= xi <=3  that's good
-0.0001 <= x4 <= 0.0001 it's also wrong  --much smaller
NOT MUCH BIGGER OR TOO MUCH SMALLER
solutions
1-mean normaliziton
2-standardisation
3-normalizition
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
