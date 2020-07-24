# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:22:12 2020

@author: Priyanka
"""

from sklearn.datasets import load_iris
data=load_iris()
print(data.keys())

import pandas as pd
new_data=pd.DataFrame(data["data"],columns=data["feature_names"])
print(data)

target=data.target
print(target)
new_data=pd.concat([new_data,pd.DataFrame(target,columns=["target"])],axis=1)
print(new_data)
X=new_data.drop(columns=["target"])
y=new_data["target"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))


from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train,y_train)
print(svm.score(X_test,y_test))

from sklearn.model_selection import KFold
kf=KFold(n_splits=5)
print(kf)

for train_index ,test_index in kf.split(X,y):
    print(train_index,test_index)

def get_scores(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)

acuracy=get_scores(LinearRegression(),X_train,X_test,y_train,y_test)
print(acuracy)

from sklearn.model_selection import StratifiedKFold

skf=StratifiedKFold(n_splits=5)
print(skf)


for train_index ,test_index in skf.split(X,y):
    print(train_index,test_index)
    

def get_score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)

score=get_score(LinearRegression(),X_train,X_test,y_train,y_test)
print(score)







