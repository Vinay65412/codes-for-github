# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 09:26:52 2020

@author: Priyanka
"""



import pandas as pd

new_data=pd.read_csv("C://Users//Priyanka//Desktop//standard stisfaction datta//santander-customer-satisfaction//train.csv",nrows=10000)
print(new_data.isnull().sum())
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np






correlation=new_data.corr(method='pearson')

def get_corelation(data,threshold):
    corr_col=set()
    correlation=data.corr(method='pearson')
    for i in range(len(correlation.columns)):
        for j in range(i):
            if correlation.iloc[i,j]>threshold:
                colname=correlation.columns[i]
                corr_col.add(colname)
    return corr_col


corr_features=get_corelation(new_data,0.85)
print(len(corr_features))



dataframe=new_data.drop(new_data[corr_features],axis=1)
print(dataframe.shape)
print(new_data.shape)


















