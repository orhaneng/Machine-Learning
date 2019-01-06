#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 13:29:52 2019

@author: omerorhan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the mall dataset
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state= 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()


#applying k-means to the mall dataset 
kmeans = KMeans(n_clusters= 5, init='k-means++', max_iter=300, n_init=10,random_state= 0)
y_kmeans = kmeans.fit_predict(x)


#visualising the clusters
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s=100, c='red',label= 'careful') 
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s=100, c='blue',label= 'standart')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s=100, c='green',label= 'target')
plt.scatter(x[y_kmeans == 3,0], x[y_kmeans == 3,1], s=100, c='cyan',label= 'careless')
plt.scatter(x[y_kmeans == 4,0], x[y_kmeans == 4,1], s=100, c='magenta',label= 'sensible') 
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow', label='centroids')
plt.title('cluster of clients')
plt.xlabel('annual income $')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()