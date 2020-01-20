#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:34:54 2020

@author: sanskruti
"""

# Problem Statement: Predict if a potential new hire is bluffing or is honest about his previous salary based on his level
# Indpendent Variables: Position and Level
# Dependent Variable: Profit ($)
# Dataset Size: 10, Use all for training the model
# Feature Selected for final model: R&D Spend And Marketing Spend($)
# Candidate Value: $160,000, Position: 6.5
# Linear Model Prediction: $330,378
# Polynomial Model Prediction: $133,259

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries_Polynomial_Regression.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""# Train Test Split  - Small Dataset, so not needed
from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

# Fitting Linear Regression To dataset - Handles feature scaling
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fitting Polynomial Regression Model 
from sklearn.preprocessing import PolynomialFeatures
# X -> Transformed to a combination of X, X^2, X^3, X^4
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the Polynonial Regression results
# Plot just X
# Plot X by creating bins
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
linear_predict = lin_reg.predict([[6.5]])
print(linear_predict)

# Predicting a new result with Polynomial Regression
poly_predict = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_predict)









# Vis



















