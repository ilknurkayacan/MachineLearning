# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:01:47 2023

@author: ilknu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv("deneyim-maas.csv",sep=";")

X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)



from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

y_predict= lr.predict(x_test)

#set sonuclarının görselleştirilmesi

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,lr.predict(x_train),color="blue")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.title("Deneyim-Maaş Train Grafiği")
plt.show()

#test sonuclarını gö

plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,lr.predict(x_train),color="blue")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.title("Deneyim-Maaş Test Grafiği")
plt.show()