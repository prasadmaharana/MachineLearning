# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:29:53 2020

@author: Prasad Maharana
"""
# to find relation between weight and height
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset

data=pd.read_csv("data.csv")

#splitting the features

X=data.iloc[:,0:1].values
Y=data.iloc[:,1].values
Y=np.reshape(Y,(15,1))

#scaling features

from sklearn.preprocessing import StandardScaler
standscx=StandardScaler()
standcy=StandardScaler()
X=standscx.fit_transform(X)
Y=standcy.fit_transform(Y)


#split the train and test set

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#create linear regression model

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,Y_train)

#predicting result

ypred=standcy.inverse_transform(linreg.predict(standscx.transform(data[['Height']])))

#plotting data and regression model on graph

plt.scatter(data['Height'],data['Weight'],color='red')
plt.plot(data['Height'],standcy.inverse_transform(linreg.predict(standscx.transform(data[['Height']]))),color='blue')
plt.show()

# r2score
import sklearn.metrics
sklearn.metrics.r2_score(data['Weight'],ypred)