import feature_eng.utils as utils
import feature_eng.data_clean as data_clean
import feature_eng.feature_eng as feature_eng
import evaluator

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import os
import gc

OUTLINER_BOUND = 0.4
OUTLINER_SAMPLE_SIZE = 80000
# Read data and feature engineering, read from pickle if possible to save time.
train_p = 'data/train_pickle'
prop_p = 'data/prop_pickle'
feature_eng_p = 'data/feature_eng_pickle'
force_read = False

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
print("Train data shape: ", train_df.shape)
#print("Outlier count: ", train_df['logerror'].sum())
# Balance training sample
print('Balancing...')
# non_outliers = train_df[train_df['logerror'] == 0].sample(
#     n=2000, replace=False, random_state=42)
# train_df = pd.concat([train_df[train_df['logerror'] == 1], non_outliers])
# print('Final training set shape: ', train_df.shape)
outlier = train_df.nlargest(2000, 'logerror')
outlier_bound = outlier['logerror'].min()
print(outlier_bound)

small = train_df.nsmallest(2000, 'logerror')
small_bound = small.loc[0, 'logerror']
print(outlier_bound)

X_train, y_train = utils.get_features_target(train_df)
X_validate, y_validate = utils.get_features_target(validate_df)

# Train classifier and predict
print('Classify...')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_validate)

print(classification_report(y_validate, y_pred))
print("outlier prediction: ", y_pred.sum())
