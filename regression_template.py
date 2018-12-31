import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
# missing data
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fitting linear regression
#create your regressor here

#prediction a new result
y_pred = regressor.redict(6.5)


# VISUALIZING the linear regression results
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Truth or bluff (regression model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# VISUALIZING the regression results(for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or bluff (polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
