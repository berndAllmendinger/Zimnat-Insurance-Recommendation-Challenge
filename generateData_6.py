# -*- coding: utf-8 -*-
#=========================================================================================================================================
# title              : generateData_6.py
# competition        : Zindi: Zimnat Insurance Recommendation Challenge
# description        : Generate Dataset train6.csv, test6.csv and sore them in folder input
# author             : Bernd Allmendinger
# ID:                : XXXX
# date               : 2020/09/14
# version            : 1.0
# usage              : start as script (for example in Syder) - I run this script in spyder
# Inputdata Folder   : Input (folder below the current working directory )
# Input files        : train.csv and .csv
# output folder      : Input folder
# output files       : train6.csv, test6.csv
# notes              :
# python_version     : 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)]
# libraries          : numpy:  1.18.1
#                      pandas:  1.0.1
#                      sklearn: 0.22.2.post1
#                      lightgbm: 2.3.1
#==========================================================================================================================================


import pandas as pd 
import numpy as np 
from sklearn import preprocessing

prodL=['P5DA', 'RIBP', '8NN1','7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']

train=pd.read_csv("input/train.csv")
test=pd.read_csv("input/test.csv")

#train['numProd']=train[prodL].apply(np.sum , axis=1)

train['numProd']=train[prodL].apply( lambda row : np.sum(row)-1, axis=1)
test['numProd']=test[prodL].apply( lambda row : np.sum(row), axis=1)

train['age']=2020-train['birth_year']
test['age']=2020-test['birth_year']

train['join_year']=pd.to_datetime(train['join_date']).dt.year
test['join_year']=pd.to_datetime(test['join_date']).dt.year

# train['join_month']=pd.to_datetime(train['join_date']).dt.month
# test['join_month']=pd.to_datetime(test['join_date']).dt.month


train['tmember']=(pd.to_datetime('03/07/2020')-pd.to_datetime(train['join_date'])).dt.days
test['tmember']=(pd.to_datetime('03/07/2020')-pd.to_datetime(test['join_date'])).dt.days

train['tuntilmember']=pd.to_datetime(train['join_date']).dt.year-train['birth_year']
test['tuntilmember']=pd.to_datetime(test['join_date']).dt.year-test['birth_year']

train['branch_codeoccupation_code'] = train['branch_code'].astype(str)+train['occupation_code']   #concat_columns(train, cols)
test['branch_codeoccupation_code'] = test['branch_code'].astype(str)+test['occupation_code']   #concat_columns(test, cols)


n=train.shape[0]
data=pd.concat([train, test], axis=0)

le_branch_codeoccupation_code = preprocessing.LabelEncoder()
data['branch_codeoccupation_code'] = le_branch_codeoccupation_code.fit_transform(data['branch_codeoccupation_code'])



le_sex = preprocessing.LabelEncoder()
data['sex'] = le_sex.fit_transform(data['sex'])

le_marital_status = preprocessing.LabelEncoder()
data['marital_status'] = le_marital_status.fit_transform(data['marital_status'])


le_branch_code = preprocessing.LabelEncoder()
data['branch_code'] = le_branch_code.fit_transform(data['branch_code'])

le_occupation_code = preprocessing.LabelEncoder()
data['occupation_code'] = le_occupation_code.fit_transform(data['occupation_code'])

le_occupation_category_code = preprocessing.LabelEncoder()
data['occupation_category_code'] = le_occupation_category_code.fit_transform(data['occupation_category_code'])

data=data.drop(['join_date'], axis=1)


train=data[:n].copy()
test=data[n:].copy()


# cerate train data set
prodD={c : i for i, c in enumerate(prodL)}
df=[]
for prod in prodD.keys():
    x=train[train[prod].isin([1])].copy()
    x['class']= prodD[prod]  
    x[prod]=0
    df.append(x)
    
df=pd.concat(df, axis=0)

    


df.to_csv("input/train6.csv",index  = False)
test.to_csv("input/test6.csv",index  = False)