'''
The below code is all about collecting the data from the csv,
pre-processing it and applying Support Vector Regression Model to
predict the outcome.
-------------------------
Author - Dibyajyoti Jena
Date - 15/10/2024
'''

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print("Independent Data:")
print(X)

print("\nDependent Data:")
print(y)

# Reshape the Dependent Variable
y = y.reshape(len(y), 1)
print("\nReshaped Dependent Data:")
print(y)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

print("\nFeature Scaling in Independent Data:")
print(X)

print("\nFeature Scaling in Dependent Data:")
print(y)

# Training the SVR model on the whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predict a new result
# Reshape is to avoid format error
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)))

# Visualizing the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)))
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()