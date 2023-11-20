import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
import joblib
import os

from tqdm.notebook import tqdm_notebook
from warnings import filterwarnings
filterwarnings('ignore')


PATH_DATA = './data'
MODEL_PATH = "./models"


model = CatBoostClassifier()  
model.load_model(os.path.join(MODEL_PATH,'model_f'))

tr_mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
tr_types = pd.read_csv(os.path.join(PATH_DATA, 'trans_types.csv'), sep=';', index_col='trans_type')

transactions = pd.read_csv(os.path.join(PATH_DATA, 'transactions.csv'), index_col='client_id')
gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'), index_col='client_id')
gender_test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'), index_col='client_id')
transactions_train = transactions.join(gender_train, how='inner')
transactions_test = transactions.join(gender_test, how='inner')
transactions_train = transactions_train.drop(['term_id'], axis=1)
transactions_test = transactions_test.drop(['term_id'],axis=1)
transactions_train = transactions_train.drop(['Unnamed: 0'], axis=1)
transactions_test = transactions_test.drop(['Unnamed: 0'],axis=1)


def calc_time_of_day(x):
    if (x >= 6) and (x < 12):
        return 'morning'
    elif (x >= 12) and (x < 18):
        return 'daytime'
    elif (x >= 18) and (x <= 23):
        return 'evening'
    else:
        return 'night'


def calc_day_type(day):
    if day == 4:
        return 'friday'
    elif (day == 5 or day == 6):
        return 'weekend'
    return 'weekday'


for df in [transactions_train, transactions_test]:
    df['day'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
    df['hour'] = df['trans_time'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
    df['night'] = ~df['hour'].between(6, 22).astype(int)
    #features from datetime
    df['ordered_day'] = df.apply(lambda row: int(row['trans_time'].split(' ')[0]),axis=1)
    df['time'] = df.apply(lambda row: row['trans_time'].split(' ')[1],axis=1)
    df['hours'] = df.apply(lambda row: int(row['trans_time'].split(' ')[1].split(':')[0]),axis=1)
    df['time_of_day'] = df.apply(lambda row: calc_time_of_day(row['hour']),axis=1)
    df['day_type'] = df.apply(lambda row: calc_day_type(row['ordered_day']%7),axis=1)

    df['week_num'] =  df['ordered_day'] // 7
    df['day_of_week'] =  df['ordered_day'] % 7
    df['month'] = df['ordered_day'] // 30 % 12
    df['day_of_month'] = df['ordered_day'] // 30 


def features_creation_advanced(x): 
    features = []
    features.append(pd.Series(x['day'].value_counts(normalize=True).add_prefix('day_')))
    features.append(pd.Series(x['hour'].value_counts(normalize=True).add_prefix('hour_')))
    features.append(pd.Series(x['night'].value_counts(normalize=True).add_prefix('night_')))
    features.append(pd.Series(x[x['amount']>0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count', 'sum', 'skew', 'kurt'])\
                                                         .add_prefix('positive_transactions_')))
    features.append(pd.Series(x[x['amount']<0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count', 'sum', 'skew', 'kurt'])\
                                                         .add_prefix('negative_transactions_')))
    
    features.append(pd.Series(x['mcc_code'].value_counts(normalize=True)).add_prefix('mcc_'))
    features.append(pd.Series(x['mcc_code'].agg(['count']).add_prefix('mcc_count')))
    
    features.append(pd.Series(x['trans_type'].value_counts(normalize=True).add_prefix('trans_')))
    features.append(pd.Series(x['trans_type'].agg(['count']).add_prefix('trans_count')))

    features.append(pd.Series(x['day_of_month'].value_counts(normalize=True).add_prefix('month_day_')))
    features.append(pd.Series(x['month'].value_counts(normalize=True).add_prefix('month_')))
    features.append(pd.Series(x['week_num'].value_counts(normalize=True).add_prefix('week_')))
    
    
    
 
    return pd.concat(features)   


data_test = transactions_test.groupby(transactions_test.index).apply(features_creation_advanced).unstack(-1)

predict= model.predict_proba(data_test)

submission = pd.DataFrame(index=data_test.index)
submission['probability'] = predict[:,1]
submission.to_csv('result.csv')
