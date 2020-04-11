# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:47:58 2020

@author: Prasad Maharana
"""
"""To apply decsion tree model"""
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")

#Creating the dependent and independent variables
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:, 2].values


#creating the regressor model
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

Y_pred=regressor.predict([[6.5]])

plt.scatter(X,Y,color="red")
plt.plot(X,regressor.predict(X),color='blue')
plt.title("Decision Tree Regression")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show

#visualizing the graph with higher resolution i.e smoother curve
X_grid=np.arange(min(X),max(X),0.001)
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,Y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title("Decision Tree Regression")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show
