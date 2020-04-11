# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:42:11 2020

@author: Prasad Maharana
"""
"""To apply polynomial regression model """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#fitting linear regression to the dataset

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)


#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,Y)

#visualizing the linear regession results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Salary prediction based on polynomial regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

x_grid=np.arange(min(X),max(X),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title("Salary prediction based on polynomial regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#predicting a new result with linear regression
lin_reg.predict([[6.5]])

lin_reg2.predict(poly_reg.fit_transform([[6.5]]))