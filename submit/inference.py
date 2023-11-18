import pandas as pd
import numpy as np
import xgboost as xgb
import re
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

PATH_DATA = './data'
MODEL_PATH = "model.pkl"

clf = joblib.load("model.pkl")


# Считываем данные
tr_mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
tr_types = pd.read_csv(os.path.join(PATH_DATA, 'trans_types.csv'), sep=';', index_col='trans_type')

transactions = pd.read_csv(os.path.join(PATH_DATA, 'transactions.csv'), index_col='client_id')
gender_test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'), index_col='client_id')
transactions_test = transactions.join(gender_test, how='inner')


# Cross-validation score (среднее значение метрики ROC AUC на тренировочных данных)
def cv_score(params, train, y_true):
    cv_res=xgb.cv(params, xgb.DMatrix(train, y_true),
                  early_stopping_rounds=10, maximize=True, 
                  num_boost_round=10000, nfold=5, stratified=True)
    index_argmax = cv_res['test-auc-mean'].argmax()
    print('Cross-validation, ROC AUC: {:.3f}+-{:.3f}, Trees: {}'.format(cv_res.loc[index_argmax]['test-auc-mean'],
                                                                        cv_res.loc[index_argmax]['test-auc-std'],
                                                                        index_argmax))

# # Построение модели + возврат результатов классификации тестовых пользователей
# def fit_predict(params, num_trees, train, test, target):
#     params['learning_rate'] = params['eta']
#     clf = xgb.train(params, xgb.DMatrix(train.values, target, feature_names=list(train.columns)), 
#                     num_boost_round=num_trees, maximize=True)
#     y_pred = clf.predict(xgb.DMatrix(test.values, feature_names=list(train.columns)))
#     submission = pd.DataFrame(index=test.index, data=y_pred, columns=['probability'])
    
#     joblib.dump(clf, MODEL_PATH)
#     return clf, submission

# # Отрисовка важности переменных. Важность переменной - количество разбиений выборки, 
# # в которых участвует данная переменная. Чем больше - тем она, вероятно, лучше 
# def draw_feature_importances(clf, top_k=10):
#     plt.figure(figsize=(10, 10))
    
#     importances = dict(sorted(clf.get_score().items(), key=lambda x: x[1])[-top_k:])
#     y_pos = np.arange(len(importances))
    
#     plt.barh(y_pos, list(importances.values()), align='center', color='green')
#     plt.yticks(y_pos, importances.keys(), fontsize=12)
#     plt.xticks(fontsize=12)
#     plt.xlabel('Feature importance', fontsize=15)
#     plt.title('Features importances, Sberbank Gender Prediction', fontsize=18)
#     plt.ylim(-0.5, len(importances) - 0.5)
#     plt.show()


params = {
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    
    'gamma': 0,
    'lambda': 0,
    'alpha': 0,
    'min_child_weight': 0,
    
    'eval_metric': 'auc',
    'objective': 'binary:logistic' ,
    'booster': 'gbtree',
    'njobs': -1,
    'tree_method': 'approx'
}

def features_creation_advanced(x): 
    features = []
    features.append(pd.Series(x['day'].value_counts(normalize=True).add_prefix('day_')))
    features.append(pd.Series(x['hour'].value_counts(normalize=True).add_prefix('hour_')))
    features.append(pd.Series(x['night'].value_counts(normalize=True).add_prefix('night_')))
    features.append(pd.Series(x[x['amount']>0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count', 'sum'])\
                                                        .add_prefix('positive_transactions_')))
    features.append(pd.Series(x[x['amount']<0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count', 'sum'])\
                                                        .add_prefix('negative_transactions_')))
    
    #added mcc
    #features.append(pd.Series(x['mcc_code'].value_counts(normalize=True).add_prefix('mcc_')))
    features.append(pd.Series(x['mcc_code'].value_counts(normalize=True)).add_prefix('mcc_'))

    return pd.concat(features)


for df in [transactions_test]:
    df['day'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
    df['hour'] = df['trans_time'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
    df['night'] = ~df['hour'].between(6, 22).astype(int)

data_test = transactions_test.groupby(transactions_test.index).apply(features_creation_advanced).unstack(-1)


# Число деревьев для XGBoost имеет смысл выятавлять по результатам на кросс-валидации 
##clf, submission = predict(params, 70, data_train, data_test, target)

def predict(params, num_trees, test, model):
     params['learning_rate'] = params['eta']
     y_pred = model.predict(xgb.DMatrix(test.values, feature_names=list(test.columns)))
     submission = pd.DataFrame(index=test.index, data=y_pred, columns=['probability'])

     return submission

submission = predict(params, 70, data_test, clf)

# Сохраняем результат моделирования
submission.to_csv('result.csv')