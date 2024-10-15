'''
This code is about collecting the data from the csv file 
and do the data pre-processing, to predict the outcome
-------------------------
Author - Dibyajyoti Jena
Date - 13/10/2024
'''

# Import the libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the data set
data_set = pd.read_csv("Data.csv")
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values
print("The Independent values:")
print(X)
print("\nThe Dependent values:")
print(y)

# Taking care of missing data
# Replacing the NaN values to the mean values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("\nAfter replacing the nan value with the mean values:")
print(X)

# Encoding the categorial data
# Encoding the Independent  Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("\nAfter encoding the independent variable:")
print(X)

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
print("\nAfter encoding the dependent variable:")
print(y)

# Splitting the dataset into Training Set and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("\nX_train data:")
print(X_train)

print("\n X_test:")
print(X_test)

print("\ny_train data:")
print(y_train)

print("\ny_test:")
print(y_test)

# Feature Scaling
# No need to Feature Scaling the Dummy Variable ie. Encoding variables
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print("\nFeature scaling the X_train:")
print(X_train)

print("\nFeature scaling the X_test:")
print(X_test)

 