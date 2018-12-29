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
dataset = pd.read_csv('Salary_Data.csv')

#missing data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test =train_test_split(x,y,test_size=1/3, random_state=0)


#fitting simple regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)