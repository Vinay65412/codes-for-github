# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:23:07 2020

@author: Priyanka
"""

import pandas as pd 
data=pd.read_csv("C://Users//Priyanka//Desktop//iris.csv")
print(data)
X=data.drop(columns=["Species_name", "Species_No","Unnamed: 0"])
y=data["Species_name"]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
skb=SelectKBest(score_func=f_classif,k=2)
fit=skb.fit(X,y)
new_data=pd.DataFrame(fit.scores_)
columns=pd.DataFrame(X.columns)
new_data=pd.concat([new_data,columns],axis=1)
new_data.columns=["specs","score"]
print(new_data)