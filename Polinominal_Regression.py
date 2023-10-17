# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:13:50 2023

@author: ilknu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("50_Startups.csv")
print(data)

X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#kayıp veri düzenleme

from sklearn.impute import SimpleImputer


imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(X[:,:-1])
X[:,:-1]=imputer.transform(X[:,:-1])

#onehotenecode işlemi

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[-1])],remainder="passthrough")

X = np.array(ct.fit_transform(X))

#train ve test setleri

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#testleri modele vermek

from sklearn.linear_model import LinearRegression

plr=LinearRegression()

plr.fit(x_train,y_train)
plr_p=plr.predict(x_test)


#polinomol regresyon modeli linearden sonra kurulur
from sklearn.preprocessing import PolynomialFeatures

pol=PolynomialFeatures(degree=4)
X_pol=pol.fit_transform(X)

plr2=LinearRegression()
plr2.fit(X_pol,y)
plr_predict=plr2.predict(X_pol)

#lineer modelin görselleşmesi
"""
plt.scatter(X,y,color="red")
plt.plot(X,plr.predict(X),color="blue")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.title("Fiyat-Kalite grafiği")
plt.show()

#ploynomal görselleşme

plt.scatter(X,y,color="red")
plt.plot(X,plr2.predict(X_pol),color="blue")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.title("Fiyat-Kalite grafiği")
plt.show()

"""













