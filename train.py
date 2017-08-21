# Train model and cross validation
# TODO(hzn):
#   1. data clean step, like drop low ratio columns, fill nan in the original prop;
#   2. training data proprocessing
#   3. model wrapper, parameter tuning
#   4. notebook api
#   5. output training records
#   6. write code to automatically add feature and see how it works

import gc
import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from features import utils
from features import feature_combine
from features import data_clean
from evaluator import Evaluator

# Configuration:
# folds number of K-Fold
FOLDS = 5

# Feature list
from features import test_feature_list
feature_list = test_feature_list.feature_list

# model
from models import XGBoost, Lightgbm
# Model = XGBoost.XGBoost
Model = Lightgbm.Lightgbm_sklearn

submit = True

# Process:
# load training data
print('Load training data...')
train, prop = utils.load_train_data()

# feature engineering
print('Feature engineering')
prop = feature_combine.feature_combine(
    prop, feature_list, False, 'features/feature_pickles/')
print(prop.shape)
# for col in prop.columns:
#     print(col)

# fill nan, inf
# prop = data_clean.clean_boolean_data(prop)
# convert string value to boolean
# prop = data_clean.drop_low_ratio_columns(prop)
prop = data_clean.clean_boolean_data(prop)
# prop = data_clean.drop_categorical_data(prop)
prop = data_clean.cat2num(prop)
print(prop.shape)

# merge transaction and prop data
df = train.merge(prop, how='left', on='parcelid')
# df.to_csv('test_df.csv')
del train; gc.collect()

# split by date
train_q1_q3, train_q4 = utils.split_by_date(df)
# train_q4.to_csv('test_train_q4.csv')
del df; gc.collect()

train_q1_q3 = data_clean.drop_training_only_column(train_q1_q3)
train_q4 = data_clean.drop_training_only_column(train_q4)
X_train_q1_q3, y_train_q1_q3 = utils.get_features_target(train_q1_q3)
X_train_q4, y_train_q4 = utils.get_features_target(train_q4)
del train_q1_q3; del train_q4; gc.collect()

# split train_q4 into k folds, each time combine k-1 folds with train_q1_q3
# to train model and validate on the left out fold
mean_errors = []
models = []
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
# X_train_q4.to_csv('test_train_q4.csv')
for train_index, validate_index in kf.split(X_train_q4):
    X_train = pd.concat([X_train_q1_q3, X_train_q4.iloc[train_index]], ignore_index=True)
    y_train = pd.concat([y_train_q1_q3, y_train_q4.iloc[train_index]], ignore_index=True)
    # TODO(hzn): add training preprocessing, like remove outliers, resampling

    X_validate = X_train_q4.iloc[validate_index]
    y_validate = y_train_q4.iloc[validate_index]

    print('training...')
    model = Model()
    model.fit(X_train, y_train)

    print('validating...')
    y_pred = model.predict(X_validate)
    if submit:
        models.append(model)
    # TODO(hzn): add output training records
    mae = Evaluator.mean_error(y_pred, y_validate)
    mean_errors.append(mae)
    print("fold mean error: ", mae)

print("average cross validation mean error", np.mean(mean_errors))

if submit:
    print("loading submission data...")
    df_test, sample = utils.load_test_data()
    print(df_test.shape)
    print(sample.shape)
    # organize test set
    df_test = df_test.merge(prop, on='parcelid', how='left')
    df_test = data_clean.drop_id_column(df_test)

    # make prediction
    print("make prediction...")
    model_preds = list(map(lambda model: model.predict(df_test), models))
    avg_pred = np.mean(model_preds, axis=0)
    print(len(avg_pred))

    # generate submission
    print("generating submission...")
    for c in sample.columns[sample.columns != 'ParcelId']:
        sample[c] = avg_pred
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sample.to_csv(
        'data/submissions/Submission_%s.csv' %time, index=False, float_format='%.4f')
    print("submission generated.")
