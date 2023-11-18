import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
import joblib

from tqdm.notebook import tqdm_notebook
from warnings import filterwarnings
filterwarnings('ignore')

tr_mcc_codes = pd.read_csv('data/mcc_codes.csv', sep=';', index_col='mcc_code')
tr_types = pd.read_csv('data/trans_types.csv', sep=';', index_col='trans_type')
transactions = pd.read_csv('data/transactions.csv', index_col='client_id')
gender_train = pd.read_csv('data/train.csv', index_col='client_id')
gender_test = pd.read_csv('data/test.csv', index_col='client_id')
transactions_train = transactions.join(gender_train, how='inner')
transactions_test = transactions.join(gender_test, how='inner')