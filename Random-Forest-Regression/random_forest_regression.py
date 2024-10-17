'''
The below code is all about collecting the data from the csv,
pre-processing it and applying Random Forest Regression Model to
predict the outcome.
-------------------------
Author - Dibyajyoti Jena
Date - 17/10/2024
'''

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset
# n_estimators is the number of trees
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)


# Predict a new value
print("Predicition value:")
print(regressor.predict([[6.5]]))

# Visualizing the Random Forest Regression result (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()