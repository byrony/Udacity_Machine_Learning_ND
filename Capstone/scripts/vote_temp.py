# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:51:19 2016

@author: caoxiang
"""
import numpy as np
import pandas as pd

result1 = pd.read_csv('submission_10fold-average-xgb_fairobj_1130.892212_2016-12-12-10-50.csv')
result1_1133 = pd.read_csv('submission_5fold-average-xgb_1133.827593_2016-12-12-01-30.csv')
result1_1112 = pd.read_csv('submission_1fold-average-xgb__2016-11-26-21-18.csv')

result2 = pd.read_csv('submission_keras_5flolds_7bags_100epochs_1132.85267681_2016-12-12-08-51.csv')
#result1 = pd.read_csv('submission_1fold-average-xgb__2016-11-26-21-18.csv')
result3 = pd.read_csv('submission_keras_5flolds_1bags_100epochs_1139.02321274_2016-12-11-21-52.csv')
result4 = pd.read_csv('submission_keras_4flolds_4bags_20epochs_1136.67827642_2016-10-27-19-31.csv')


result_id =  result1['id'].values
result_pred = (0.25 * result1['loss'].values 
                + 0.15 * result1_1133['loss'].values 
                + 0.15 * result1_1112['loss'].values 
                + 0.25* (7*result2['loss'].values + result3['loss'].values)/8 
                + 0.25* result4['loss'].values)


df = pd.DataFrame({'id': result_id, 'loss': result_pred})
df.to_csv('xgboost_nn_ensemble7.csv', index = False)