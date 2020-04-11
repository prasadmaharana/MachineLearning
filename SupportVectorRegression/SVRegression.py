# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:16:01 2020

@author: Prasad Maharana
"""
""" To apply SVR or Support Vector regression"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2:].values

#apply feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y)


#fitting the SVR to the dataset

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)

#predictring the results
y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualizaing the SVR results
plt.scatter(X,Y,color="red")
plt.plot(X,regressor.predict(X),color="green")
plt.title("SVR")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
