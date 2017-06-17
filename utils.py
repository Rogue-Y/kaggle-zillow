import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# utility functions
def plot_score(y, y_hat):
    return (np.mean(abs(y-y_hat)))

def load_data():
    train = pd.read_csv('train_2016.csv')
    prop = pd.read_csv('properties_2016.csv')
    return train.merge(prop, how='left', on='parcelid')

def train_valid_split(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def predict(predictor, train_cols):
    sample = pd.read_csv('sample_submission.csv')
    prop = pd.read_csv('properties_2016.csv')
    sample['parcelid'] = sample['ParcelId']
    df_test = sample.merge(prop, on='parcelid', how='left')
    x_test = df_test[train_cols]
    for c in x_test.dtypes[x_test.dtypes == object].index.values:
        x_test[c] = (x_test[c] == True)
    predictor.reset_parameter({"num_threads":1})
    p_test = predictor.predict(x_test)
    sub = pd.read_csv('sample_submission.csv')
    for c in sub.columns[sub.columns != 'ParcelId']:
        sub[c] = p_test

    sub.to_csv('lgb_starter.csv', index=False, float_format='%.4f')
