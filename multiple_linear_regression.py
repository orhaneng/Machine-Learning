import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('venv/50_Startups.csv')

# missing data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

x = x[:, 1:]

# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#fitting multiple linear regession to the training set
from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test variables
y_pred = regressor.predict(x_test)

import statsmodels.formula.api as sm
x = np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

x_opt = x[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())


x_opt = x[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())



x_opt = x[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())



x_opt = x[:,[0,3,5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())

x_opt = x[:,[0,3]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())
