import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

# missing data
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fitting linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

# fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
print(x_poly)

# VISUALIZING the linear regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or bluff (linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# VISUALIZING the polynomial regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or bluff (polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

test = np.array([6.5,])

test = test.reshape(1, -1)

print(lin_reg.predict(test))


print(lin_reg2.predict(poly_reg.fit_transform(test)))