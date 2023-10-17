# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:47:06 2023

@author: ilknur
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
data=pd.read_csv("Data2.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#missingvalue


#train-test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#modelling

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


lr=LinearRegression()
pl=PolynomialFeatures(degree=4)
x_pol=pl.fit_transform(X_train)
lr.fit(x_pol,y_train)


y_pred=lr.predict(pl.transform(X_test))
#tahmini ve gerçek değerleri karşılaştıracak
ComparePoly=np.concatenate((y_pred.reshape(len(y_pred,),1),y_test.reshape(len(y_test),1)),1)

#RSqured değerini hesapla

from sklearn.metrics import r2_score
r2_score_polylineer=r2_score(y_test,y_pred)


