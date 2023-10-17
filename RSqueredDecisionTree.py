# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:00:32 2023

@author: ilknu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Data2.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=0)

dt.fit(X_train, y_train)

y_pred=dt.predict(X_test)
#tahmini ve gerçek değerleri karşılaştıracak
CompareDecision=np.concatenate((y_pred.reshape(len(y_pred,),1),y_test.reshape(len(y_test),1)),1)


#RSqured değerini hesapla

from sklearn.metrics import r2_score
r2_score_decision=r2_score(y_test,y_pred)