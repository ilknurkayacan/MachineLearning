# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:17:30 2023

@author: ilknu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri yükleme
data=pd.read_csv("kalite-fiyat.csv")

X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#train-test set 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#random forest uygulama 

from sklearn.ensemble import  RandomForestRegressor

rf=RandomForestRegressor(random_state=0,n_estimators=10)

rf.fit(X,y)
y_tahmin=rf.predict([[5.5]])

x_grid=np.arange(min(X),max(X),0.1)
x_grid=x_grid.reshape((len(x_grid),1))

plt.scatter(X, y, color="red")
plt.plot(x_grid,rf.predict(x_grid),color="blue")
plt.title("Fiyat-Kalite Grafiği (Polynomial) Degree=4")
plt.xlabel("Kalite")
plt.ylabel("Fiyat")
plt.show()