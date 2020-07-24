# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:55:22 2020

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
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
#for backward selection
sbs=SFS(LinearRegression(),k_features=2)
bckwrd=sbs.fit(X,y)

print(bckwrd.k_feature_names_)



# for forward selecion
sfs=SFS(LinearRegression(),k_features=2,forward=True)
forward_slection=sfs.fit_transform(X,y)

print(sfs.k_feature_names_)
















