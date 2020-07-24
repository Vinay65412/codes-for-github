# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:27:00 2020

@author: Priyanka
"""


import pandas as pd 
data=pd.read_csv("C://Users//Priyanka//Desktop//iris.csv")
print(data)
X=data.drop(columns=["Species_name", "Species_No","Unnamed: 0"])
y=data["Species_name"]
from sklearn.ensemble import ExtraTreesClassifier
best_feat=ExtraTreesClassifier()
model=best_feat.fit(X,y)

feature_imp=model.feature_importances_
print(feature_imp)
model_feat=pd.Series(feature_imp,index=X.columns)
import matplotlib.pyplot as plt
model_feat.plot(kind="barh")
plt.show()