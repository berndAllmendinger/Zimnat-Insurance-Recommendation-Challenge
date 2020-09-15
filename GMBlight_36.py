#=========================================================================================================================================
# title              : GMBlight_36.py
# competition        : Zindi: Zimnat Insurance Recommendation Challenge
# description        : Read the data train6.csv and test6.csv from input folder, fitt a lightgbm model and generate prediction for test
# author             : Bernd Allmendinger
# ID:                : xxxx
# date               : 2020/09/14
# version            : 1.0
# usage              : start as script (for example in Syder) - I run this script in spyder
# Inputdata Folder   : Input (folder below the current working directory )
# Input files        : train6.csv, test6.csv
# output folder      : submit (folder below the current working directory )
# output files       : submit_lgb_36.csv
# notes              :
# python_version     : 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)]
# libraries          : numpy:  1.18.1
#                      pandas:  1.0.1
#                      sklearn: 0.22.2.post1
#                      lightgbm: 2.3.1
#==========================================================================================================================================


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 40)
#import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from itertools import combinations


np.random.seed(2019)
Version = 36

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """
    Multi class version of Logarithmic Loss metric.
        
    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]
    
    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)
    
    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]
    
    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota


train = pd.read_csv('input/train6.csv')
test = pd.read_csv('input/test6.csv')


features=['sex', 'marital_status', 'birth_year', 'branch_code', 
       'occupation_code',  'P5DA', 'RIBP', '8NN1',
       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3',
       'numProd', 'join_year', 'tmember', 'branch_codeoccupation_code']

catfeatures=['sex', 'marital_status',  'branch_code','branch_codeoccupation_code'
       'occupation_code', 'occupation_category_code']


catfeaturesind = [ind for ind, f in enumerate(features) if f in catfeatures]


prodL=['P5DA', 'RIBP', '8NN1','7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']
colpredL=['pred'+c for c in prodL]



cmbs = list(combinations(prodL, 2))

# add new features based on combinations/interactions
for cols in cmbs:
    train["".join(cols)] = train[cols[0]]+train[cols[1]]   #concat_columns(train, cols)
    test["".join(cols)] = test[cols[0]]+test[cols[1]]   #concat_columns(test, cols)

comb_col=[]
for cols in cmbs:
    comb_col.append("".join(cols))

features += comb_col


train['tmembernumProd'] = train['tmember']*train['numProd']   #concat_columns(train, cols)
test['tmembernumProd'] = test['tmember']*test['numProd']   #concat_columns(test, cols)
features += ['tmembernumProd'] 



numclass=len(train['class'].unique())

params ={
 'boosting_type': 'gbdt',
 'learning_rate': 0.001,
 'objective': 'multiclass',
 'num_class': numclass,
 'metric': 'multi_logloss',
 'num_leaves': 50, 
 'bagging_freq': 5, 
 'min_child_samples': 23, 
 'min_data_in_leaf': 20,  
 'feature_fraction': 0.4, 
 'bagging_fraction': 0.9, 

 'min_sum_hessian_in_leaf': 10,
 'max_depth' : 8 
}


x_train=train[features]
x_test=test[features]
x_train.head()
y_train = train['class']


nfolds=5

gkf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=2022) #GroupKFold(n_splits=nfolds)
kfold_split = gkf.split(x_train, y_train)
foldslist = list(kfold_split)    

#dtrain = lgb.Dataset(x_train.values,  y_train.values, free_raw_data=False)
dtrain = lgb.Dataset(x_train,  y_train, free_raw_data=False, categorical_feature=catfeaturesind)

gbm = lgb.cv(params,
                early_stopping_rounds=100,
                #nfold =5,
                folds= (i for i in foldslist),
                stratified=True,
                verbose_eval=1,
                #feval=lgb_mae,
                train_set =dtrain,
                num_boost_round=15000)

#best_rounds=gbm.best_iteration
best_rounds=np.argmin(gbm['multi_logloss-mean'])+1
best_score = np.min(gbm['multi_logloss-mean'])
print ("best score %8.5f best round %i" % (best_score, best_rounds))


##############################
gbm = lgb.train(params, dtrain,  num_boost_round=best_rounds) 
plot=lgb.plot_importance(gbm, max_num_features=40, figsize=(8, 40))

        

preds = gbm.predict(x_test)


dfpred=pd.DataFrame(preds, columns=prodL)

dfpred=dfpred.reset_index(drop=True)
x_test=x_test.reset_index(drop=True)

for i in range(dfpred.shape[0]):
   # sump=0
    for c in dfpred.columns:
        if x_test.loc[i, c]==1:
            dfpred.loc[i, c]=1
        #sump += dfpred.loc[i, c]  
     
pred=pd.concat([test['ID'], dfpred], axis=1)

sub=pred.set_index('ID').stack().reset_index()

sub.columns=['ID', 'PCODE', 'Label']
sub['ID X PCODE']=sub['ID']+' X '+sub['PCODE']

sub[['ID X PCODE',  'Label']].to_csv('submit/submit_lgb_'+str(Version)+'.csv',index  = False)