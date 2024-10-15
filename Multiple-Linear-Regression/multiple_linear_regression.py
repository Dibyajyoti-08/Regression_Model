'''
The below code is all about collecting the data from the csv,
pre-processing it and applying Multiple Linear Regression Model to
predict the outcome.
-------------------------
Author - Dibyajyoti Jena
Date - 13/10/2024
'''

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding  categorial data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print('After encoding categorial data:')
print(X)
print('\n')

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))