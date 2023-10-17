# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:24:56 2023

@author: ilknu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yükleme
data=pd.read_csv("Data2.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#kayıp veri
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(X[:,:-1])
X[:,:-1]=imputer.transform(X[:,:-1])

#train-test set
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#model uygula
from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(X_train,y_train)

y_pred=ml.predict(X_test)
#tahmini ve gerçek değerleri karşılaştıracak
CompareMulti=np.concatenate((y_pred.reshape(len(y_pred,),1),y_test.reshape(len(y_test),1)),1)

#RSqured değerini hesapla

from sklearn.metrics import r2_score
r2_score_multilineer=r2_score(y_test,y_pred)






















