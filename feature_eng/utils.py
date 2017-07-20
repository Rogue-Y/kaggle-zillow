import feature_eng.data_clean as data_clean
import feature_eng.feature_eng as feature_eng

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# utility functions
def plot_score(y, y_hat):
    return (np.mean(abs(y-y_hat)))

def load_train_data():
    """ Load data and join transaction data with properties data.
        Returns:
            (train_df, properties_df, joined_df)
    """
    train = pd.read_csv('data/train_2016_v2.csv')
    prop = pd.read_csv('data/properties_2016.csv')
    # df = train.merge(prop, how='left', on='parcelid')
    return (train, prop)

def load_test_data():
    """ Load data and join trasaction data with properties data.
        Returns:
            (joined_test_df, sample_submission_df)
    """
    sample = pd.read_csv('data/sample_submission.csv')
    # sample submission use "ParcelId" instead of "parcelid"
    test = sample.rename(index=str, columns={'ParcelId': 'parcelid'})
    # drop the month columns in sample submission
    test = test.drop(['201610', '201611', '201612', '201710', '201711', '201712'], axis=1)
    return (test, sample)

def train_valid_split(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def get_features_target(df):
    """ Get features dataframe, and target column
        Call clean data and drop column in data_clean.py function before use
        this method.
        Returns:
            (X, y)
    """
    # logerror is the target column
    # transactiondate only available in training data
    return (df.drop(['logerror','transactiondate'], axis=1), df['logerror'])

def split_by_date(df, split_date = '2016-10-01'):
    """ Split the transaction data into two part, those before split_date as
        training set, those after as test set.
        Returns:
            (train_df, test_df)
    """
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    # 82249 rows
    train_df = df[df['transactiondate'] < split_date]
    # 8562 rows
    test_df = df[df['transactiondate'] >= split_date]
    return (train_df, test_df)

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

def get_train_test_sets():
    """ Get the training and testing set: now split by 2016-10-01
        transactions before this date serve as training data; those after as
        test data.
        Returns:
            (X_train, y_train, X_test, y_test)
    """
    print('Loading data ...')
    train, prop = load_train_data()

    print('Cleaning data and feature engineering...')
    prop_df = feature_eng.add_missing_value_count(prop)
    prop_df = data_clean.clean_categorical_data(prop_df)

    # Subset with transaction info
    df = train.merge(prop_df, how='left', on='parcelid')
    df = feature_eng.convert_year_build_to_age(df)
    df = data_clean.drop_low_ratio_columns(df)
    df = data_clean.clean_boolean_data(df)
    df = data_clean.drop_id_column(df)
    df = data_clean.fillna(df)

    print("Spliting data into training and testing...")
    train_df, test_df = split_by_date(df)
    # 82249 rows
    X_train, y_train = get_features_target(train_df)
    # 8562 rows
    X_test, y_test = get_features_target(test_df)

    print("Done")
    return (X_train, y_train, X_test, y_test)
