#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:08:40 2018

@author: omerorhan
"""

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
dataset = pd.read_csv('Social_Network_Ads.csv')

#missing data
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

#splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train , y_test =train_test_split(x,y,test_size=0.25, random_state=0)

'''
feature scalling
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
    

#prediction the test set results 
y_pred = classifier.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


#visualising the training test resuts
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:,0].min() -1 ,stop= x_set[:,0].max()+1, step=0.01),
np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1, step=0.01))
plt.contour(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,
            cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0], x_set[y_set == j,1], c= ListedColormap(('red','green'))(i),label = j)
plt.title('classification (training set)')
plt.xlabel('age') 
plt.ylabel('estimated salary')
plt.legend()
plt.show() 


#visualising the test test resuts
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:,0].min() -1 ,stop= x_set[:,0].max()+1, step=0.01),
np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1, step=0.01))
plt.contour(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,
            cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0], x_set[y_set == j,1], c= ListedColormap(('red','green'))(i),label = j)
plt.title('classification (training set)')
plt.xlabel('age') 
plt.ylabel('estimated salary')
plt.legend()
plt.show() 

