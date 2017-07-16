import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# utility functions
def plot_score(y, y_hat):
    return (np.mean(abs(y-y_hat)))

def load_train_data():
    """ Load data and join trasaction data with properties data.
        Returns:
            (train_df, properties_df, joined_df)
    """
    train = pd.read_csv('data/train_2016.csv')
    prop = pd.read_csv('data/properties_2016.csv')
    return (train, prop, train.merge(prop, how='left', on='parcelid'))

def load_test_data(prop):
    """ Load data and join trasaction data with properties data.
        Params:
            prop - properties dataframe
        Returns:
            (joined_test_df, sample_submission_df)
    """
    sample = pd.read_csv('data/sample_submission.csv')
    test = sample.rename(index=str, columns={'ParcelId': 'parcelid'})
    return (test.merge(prop, on='parcelid', how='left'), sample)

def train_valid_split(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def get_features_target(df):
    """ Get features dataframe, and target column
        Call clean data and drop column in data_clean.py function before use
        this method.
        Returns:
            (X, y)
    """
    return (df.drop('logerror', axis=1), df['logerror'])

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

def predict(predictor, X_test, sample, suffix=''):
    """ Predict on test set and write to a csv
        Params:
            predictor - the predictor, using lightgbm now
            X_test - the test features dataframe
            sample - sample_submission dataframe
            suffix - suffix of output file
    """
    predictor.reset_parameter({"num_threads":1})
    p_test = predictor.predict(X_test)
    for c in sample.columns[sample.columns != 'ParcelId']:
        sample[c] = p_test

    sample.to_csv('data/lgb_starter'+suffix+'.csv', index=False, float_format='%.4f')
