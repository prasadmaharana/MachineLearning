# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:48:28 2020

@author: Prasad Maharana
"""

#To predict pricing of house based on multiple predictors using Multiple Linear Regression

#importing required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#importing dataset
data=pd.read_csv("kc_house_data.csv")

#check for missing values
missing=pd.isnull(data)
data[missing]   #filter on missing values

#defining independent and dependent features
data=data.drop('id',axis=1) #dropping ID column
data=data.drop('date',axis=1) #dropping date column

X=data.iloc[:, 1:19].values     #apartment features
Y=data.iloc[:,0:1].values       #price (dependent)

#scaling features
from sklearn.preprocessing import StandardScaler
StanScX=StandardScaler() #for features
 #Since there is one dependent variable, I chose not to scale. (Increases number of ops)
 
X=StanScX.fit_transform(X)

#appending ones to the matrix of features
X=np.append(arr=np.ones((21613,1)).astype(int),values=X,axis=1)

#backward elimation method to remove unwanted predictors

import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

def BackwardEliminationMethod(x,SL):
    VarCount=len(x[0])
    for i in range (0,VarCount):
        regressorOLS=sm.OLS(Y,x).fit()
        MaxVarP=max(regressorOLS.pvalues).astype(float)
        if MaxVarP > SL:
            for j in range (0,VarCount-i):
                if regressorOLS.pvalues[j].astype(float)==MaxVarP:
                    x=np.delete(x,j,1)
    regressorOLS.summary()
    return x
    
SL=0.05
X_optimized=BackwardEliminationMethod(X_opt,SL)
regressorOLS=sm.OLS(Y,X_opt).fit()
regressorOLS.summary()

#still one variable has p-value more than 0.05
X_opt=X_optimized[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16]]

#removed var x5 and refitted the OLS 
regressorOLS=sm.OLS(Y,X_opt).fit()
regressorOLS.summary()

from  sklearn.linear_model import LinearRegression
Linreg=LinearRegression()
Linreg.fit(X_opt,Y)

y_pred=Linreg.predict(X_opt)

#r2score

from sklearn.metrics import r2_score
r2_score(Y,y_pred)

plt.plot(X_opt,)

