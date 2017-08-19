import feature_eng.utils as utils
import feature_eng.data_clean as data_clean
import feature_eng.feature_eng as feature_eng
from evaluator import Evaluator

import math
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

import os
import gc

OUTLINER_BOUND = 0.4
OUTLINER_SAMPLE_SIZE = 80000
# Read data and feature engineering, read from pickle if possible to save time.
train_p = 'data/train_pickle'
prop_p = 'data/prop_pickle'
feature_eng_p = 'data/feature_eng_pickle'
force_read = False

xgboost_p = 'data/models/xgboost_pickle'
force_train = True

print('Loading data...')
if not force_read and os.path.exists(train_p):
    train = pd.read_pickle(train_p)
else:
    train = utils.load_transaction_data()
    train.to_pickle(train_p)


if not force_read and os.path.exists(feature_eng_p):
    prop = pd.read_pickle(feature_eng_p)
else:
    if not force_read and os.path.exists(prop_p):
        prop = pd.read_pickle(prop_p)
    else:
        prop = utils.load_properties_data()
        prop.to_pickle(prop_p)
    print('Processing data...')
    prop = feature_eng.add_missing_column_boolean(prop)
    prop = feature_eng.add_missing_value_count(prop)
    prop = data_clean.drop_low_ratio_columns(prop)
    prop = feature_eng.add_features(prop)
    prop = data_clean.encode_categorical_data(prop)
    prop = feature_eng.add_before_1900_column(prop)
    prop = data_clean.clean_boolean_data(prop)
    prop = data_clean.clean_strange_value(prop)
    prop.to_pickle(feature_eng_p)

df = train.merge(prop, how='left', on='parcelid')
del train; gc.collect()

# Change label for classification
#df['logerror'] = abs(df['logerror']) > OUTLINER_BOUND
# Split train and validate data
train_df, validate_df = utils.split_by_date(df)
train_df = data_clean.drop_training_only_column(train_df)
validate_df = data_clean.drop_training_only_column(validate_df)
print("Train data shape before resample: ", train_df.shape)

# Resample the training set
train_df['logerror_cut'] = pd.cut(train_df['logerror'], 11, labels=False)
resamples = []
for i in range(11):
    dist = pow(max(i - 5, 1), 5)
    resample = train_df[train_df['logerror_cut'] == i].sample(
        math.ceil(20000 / dist), replace = True, random_state = 42)
    resamples.append(resample)
train_df = pd.concat(resamples, copy=False)
print("Train data shape after resample: ", train_df.shape)
train_df.drop('logerror_cut', axis=1, inplace=True)
print("Train data shape after resample: ", train_df.shape)

#print("Outlier count: ", train_df['logerror'].sum())
# Balance training sample
# print('Balancing...')
# # non_outliers = train_df[train_df['logerror'] == 0].sample(
# #     n=2000, replace=False, random_state=42)
# # train_df = pd.concat([train_df[train_df['logerror'] == 1], non_outliers])
# # print('Final training set shape: ', train_df.shape)
# outlier = train_df.nlargest(2000, 'logerror')
# outlier_bound = outlier['logerror'].min()
# print(outlier_bound)
#
# small = train_df.nsmallest(2000, 'logerror')
# small_bound = small.loc[0, 'logerror']
# print(outlier_bound)

X_train, y_train = utils.get_features_target(train_df)
X_validate, y_validate = utils.get_features_target(validate_df)

# Train classifier and predict
print('Classify...')
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_validate)

# xgboost
if not force_train and os.path.exists(xgboost_p):
    with open(xgboost_p, 'rb') as xgboost_p_handler:
        model = pickle.load(xgboost_p_handler)
else:
    print("\nSetting up data for XGBoost ...")
    # xgboost params
    xgb_params = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': 0.4,
        # 'base_score': y_mean,
        'silent': 1
    }

    d_train = xgb.DMatrix(X_train, y_train)

    num_boost_rounds = 250
    print("num_boost_rounds="+str(num_boost_rounds))

    # train model
    print( "\nTraining XGBoost ...")
    model = xgb.train(dict(xgb_params, silent=1), d_train, num_boost_round=num_boost_rounds)
    with open(xgboost_p, 'wb') as xgboost_p_handler:
        pickle.dump(model, xgboost_p_handler)

print( "\nPredicting with XGBoost ...")
d_validate = xgb.DMatrix(X_validate)
y_pred = model.predict(d_validate)

# print(classification_report(y_validate, y_pred))
print("average mean error: ", Evaluator.mean_error(y_pred, y_validate))
print("prediction summary: ", pd.Series(y_pred).describe())
