# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 10:21:38 2020

@author: Priyanka
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
data=pd.read_csv("C://Users//Priyanka//Desktop//standard stisfaction datta//santander-customer-satisfaction//train.csv",nrows=10000)
print(data.isnull().sum())
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


X=data.drop(columns=["TARGET"])
Y=data["TARGET"]


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)

print(X_train.shape,X_test.shape)
constant_filter=VarianceThreshold(threshold=0)
constant_filter.fit(X_train)

print(len(X_train.columns[constant_filter.get_support()]))


constant_columns=[column for column in X_train.columns 
                 if column not in X_train.columns[constant_filter.get_support()] ]
print(constant_columns)

print(len(constant_columns))


X_train_no_const=constant_filter.transform(X_train)
X_test_no_const=constant_filter.transform(X_test)

print(X_train_no_const.shape,X_test_no_const.shape)

#X_train_no_const=X_train.drop(labels=constant_columns,axis=1,inplace=True)

#X_test_no_const=X_test.drop(labels=constant_columns,axis=1,inplace=True)


#removing the quasi constant



quasi_const=VarianceThreshold(threshold=0.01)
quasi_const.fit(X_train_no_const)


print(quasi_const.get_support().sum())



X_train_no_qconst=quasi_const.transform(X_train_no_const)
X_test_no_qconst=quasi_const.transform(X_test_no_const)

print(X_train_no_qconst.shape,X_test_no_qconst.shape)

#remove  the duplicate feature 



X_train_duplicate=X_train_no_qconst.T
X_test_duplicate=X_test_no_qconst.T


#converted into dataframe for ----


X_train_duplicate=pd.DataFrame(X_train_duplicate)
X_test_duplicate=pd.DataFrame(X_test_duplicate)
print(X_train_duplicate.shape,X_train_duplicate.shape)
#check the number of duplicate feature
print(X_train_duplicate.duplicated().sum())


X_train_unique=X_train_duplicate.drop_duplicates().T
X_test_unique=X_test_duplicate.drop_duplicates().T
print(X_train_unique.shape,X_test_unique.shape)

"""
from sklearn.feature_selection import chi2

bestfeatures=SelectKBest(score_func=chi2,k=3)
fit=bestfeatures.fit(X,Y)
#fit=bestfeatures.transform(X,y)


df_score=pd.DataFrame(fit.scores)
df_columns=pd.DataFrame(X.columns)


feature_score=pd.concat([df_score,df_columns],axis=1)
feature_score.columns=["specs","score"]

print(feature_score)

print(feature_score.nlargest(10))
"""

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model= ExtraTreesClassifier()
model.fit(X,Y)

print(model.feature_importances_)



feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(10).plot(kind="barh")
plt.show()





























