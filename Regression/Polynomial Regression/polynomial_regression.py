# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Training the Polynomial model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_reg = PolynomialFeatures(degree=4)
x_poly = polynomial_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the Linear Regression results
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial results
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg_2.predict(polynomial_reg.fit_transform(x)), color="blue")
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, lin_reg_2.predict(polynomial_reg.fit_transform(x_grid)), color="blue")
plt.title("Higher Resolution (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(polynomial_reg.fit_transform([[6.5]])))

