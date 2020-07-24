# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:49:25 2020

@author: Priyanka
"""

from sklearn.datasets import load_iris


data=load_iris()
print(data)


import pandas as pd
new_data=pd.DataFrame(data["data"],columns=data["feature_names"])
print(new_data)

target=data.target
print(target)

new_data=pd.concat([new_data,pd.DataFrame(target,columns=["target"])],axis=1)
print(new_data)

new_data.columns=["sepal_length" , "sepal_width" ,"petal_length"," petal_width" , "target"]
print(new_data)

X=new_data.drop(columns=["target"])
y=new_data["target"]
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
skb=SelectKBest(score_func=f_regression,k=2)
fit=skb.fit(X,y)
print(fit)



data=pd.DataFrame(fit.scores_)
data_columns=pd.DataFrame(X.columns)
new_data=pd.concat([data,data_columns],axis=1)
new_data.columns=["look","score"]
print(new_data)
#print(data)




















