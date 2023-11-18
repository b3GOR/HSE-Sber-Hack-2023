import joblib
import pandas as pd


model = joblib.load('submission/models/model.pkl')

new_data = pd.read_csv('data/test_data.csv')

new_data = transactions_test.groupby(transactions_test.index)\
                            .progress_apply(features_creation_advanced).unstack(-1)
