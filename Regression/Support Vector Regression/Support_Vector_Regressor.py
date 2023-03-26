# Predict wage for 6.5 years of experience
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)

# Feature Scaling
# Different means for level and salary so different sc scaler
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#print(x)
#print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(x, y)

# Predicting a new result
# Putting 6.5 years in *[[tuple]]* and reverse scaling to get salary in normal, reshape for not get any errors
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))
#print(y_pred)


# Visualising the SVR results
# Reverse transform to get better plot labels scatter
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color="blue")
plt.title("Support Vector Regressor")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
x_inv = sc_x.inverse_transform(x)
y_inv = sc_y.inverse_transform(y)
x_grid = np.arange(min(x_inv), max(x_inv), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x_inv, y_inv, color='red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1,1)), color="blue")
plt.title("Support Vector Regressor - High Resolution")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
