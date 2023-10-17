# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:49:07 2023

@author: ilknu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yükleme
data=pd.read_csv("50_Startups.csv")
print(data)

X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#kayıp data
from sklearn.impute import SimpleImputer

impute=SimpleImputer(missing_values=np.nan,strategy="mean")
impute.fit(X[:,:-1])
X[:,:-1]=impute.transform(X[:,:-1])


#kategorik verileri numeric veriye çevirme

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[-1])],remainder="passthrough")

X = np.array(ct.fit_transform(X))

#feature sacling yapmak
#ancal muştpile lineer regresyonda feature scalinge gerek yoktur
#Train ve Test Setleri

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)



#train ile öğrenmesi
from sklearn.linear_model import LinearRegression

mlr=LinearRegression()
mlr.fit(x_train,y_train)

#modelin test edilmesi
y_predict=mlr.predict(x_test)


plt.scatter(X,y,color="red")
plt.plot(X,mlr.predict(X),color="blue")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.title("Fiyat-Kalite grafiği")
plt.show()









