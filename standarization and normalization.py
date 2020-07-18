# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:26:05 2020

@author: Priyanka
"""
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split 
data=pd.read_csv("C://Users//Priyanka//Desktop//forc.csv")
print (data)
print(data.describe())
X=data[["age","mileage"]]
y=data["sellprice"]
print(X.shape)
print(y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("")
# bellow step we do for only covert the the mean and standard into 0 and 1
#this for is only done(featuring scaling after the data cleanning )
#sd and mean of only traning data and testing data (independent variable not for dependent varialbe)
sc=StandardScaler()
sc.fit(X_train)
sc.fit(X_test)
print(sc.mean_)
print(sc.scale_)
# now tranfrom the scalled data into xtrain and x test for further process
X_test_sd=sc.transform(X_test)
X_train_sd=sc.transform(X_train)
print(X_train_sd)
X_train_sd=pd.DataFrame(X_train_sd,columns=["age","mileage"])
X_test_sd=pd.DataFrame(X_test_sd,columns=["age","mileage"])
print(X_train_sd.describe().round(0))
print(data.describe().round(2 ))


"""for min and max================================================================================= """

min_data=MinMaxScaler()
min_data.fit(X_train)
"""now for the transformation"""

X_train_min_data=min_data.transform(X_train)
X_test_min_data=min_data.transform(X_test)

X_train_min=pd.DataFrame(X_train_min_data,columns=["age","mileage"])
X_test_min=pd.DataFrame(X_test_min_data,columns=["age","mileage"])
print(X_train_min)
print(X_test_min)






