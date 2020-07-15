# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:05:44 2020

@author: Priyanka
"""
#from dataset import load_dataset 
import scipy.stats as stats
import seaborn as sns 
import pandas as pd
import numpy as np
data=sns.load_dataset("tips")
print(data) 
print(data.head(5))

 #its a cross check way  between two categorical variable with there categories 


data_table=pd.crosstab(data["sex"],data["smoker"])
print(data_table) 


#just tranfer the value into observed values from the data_table
#observed value  which is requier when we find chisquare statistic test

observed_value=data_table.values
print("observed value:   ",observed_value)

#  for getting the expected value  from the observed data_table
val=stats.chi2_contingency(data_table)
print("represent the expected value:  ",val)


expected_value=val[3]
print("expected_value  :",expected_value)


# calculate the degree of freedom(value that have freedom to vary in the data sample )
no_rows=2
no_columns=2
dof=(no_rows-1)*(no_columns-1)
print("degree of freedom:  ",dof)

alpha=0.05


#calculate chisqure statistics test 
from scipy.stats  import chi2
chi_square_test=sum([(o-e)**2/e for o,e in zip(observed_value,expected_value)]) 
print("chi_square:  ",chi_square_statistics_test)



#critical value  is a point in the data set which we compared with test statistics to check whether nnull hypothesis is rejected or not  
# Here q is The probability to  expressed significance

critical_value=chi2.ppf(q=1-alpha,df=dof)
print("critical_value:  ",critical_value)



#p_value is just like evidence for selecting the alternate hypothssis 

p_value=1-chi2.cdf(x=chi_square_test,df=dof)
print("p_value:  ",p_value)

# if chi_square_test>=critical_vlaue may me it give error
#where more than one element is ambigiuous
#on such case we use condition like a.any() and a.all()
#where a=(chi_square_test>=critical_value)

if (chi_square_test>=critical_value).any():
    print("Ho is rejected  -----containing two categorical variable")
else:
  print("H1 is retain-----contaning two categorical variable")    


#we can also compare p_values for the acceptance and rejection of the hypothesis test





























