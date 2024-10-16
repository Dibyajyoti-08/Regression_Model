'''
The below code is all about collecting the data from the csv,
pre-processing it and applying Decision Tree Regression Model to
predict the outcome.
-------------------------
Author - Dibyajyoti Jena
Date - 16/10/2024
'''

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
print("Prediction Value:")
print(regressor.predict([[6.5]]))

# Visualising the Decision Tree Regression results(higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or Bluff (Decision Regression Tree)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()



