# Train model and cross validation
# TODO(hzn):
#   1. data clean step, like drop low ratio columns, fill nan in the original prop;
#   2. training data proprocessing
#  *3. model wrapper, model parameter tuning
#   4. notebook api
#   5. output training records
#   6. write code to automatically add feature and see how it works
#   7. move configuration to a standalone file, and read it from cmd line

import gc
import datetime
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from features import utils
from features import feature_combine
from features import data_clean
from evaluator import Evaluator
import config

# Get configuration
# parser to parse cmd line option
parser = OptionParser()
# add options to parser, currently only config file
parser.add_option('-c', '--config', action='store', type='string', dest='config_file')
# parse cmd line arguments
(options, args) = parser.parse_args()

config_file = options.config_file
# default to test config
if not config_file:
    config_file = 'test_config'

# Configuration:
config_dict = getattr(config, config_file)

# Mandatory configurations:
# Feature list
feature_list = config_dict['feature_list']
# model
Model = config_dict['model']

# Optional configurations:
# folds number of K-Fold
FOLDS = config_dict['folds'] if 'folds' in config_dict else 5
# if record training
record = config_dict['record'] if 'record' in config_dict else False
# if generate submission or not
submit = config_dict['submit'] if 'submit' in config_dict else False

# Helper functions:
def record_train(train_recorder, y_train, y_train_pred, y_valid, y_valid_pred):
    y_train = pd.Series(y_train)
    y_train_pred = pd.Series(y_train_pred)
    y_valid = pd.Series(y_valid)
    y_valid_pred = pd.Series(y_valid_pred)
    mean_train_error = Evaluator.mean_error(y_train, y_train_pred)
    mean_valid_error = Evaluator.mean_error(y_valid, y_valid_pred)
    train_recorder.write('Train error: ' + str(mean_train_error) + '\n')
    train_recorder.write('Validation error: ' + str(mean_valid_error) + '\n')
    train_recorder.write('\nTrain Stats \n')
    train_recorder.write('Train label stats: ' + y_train.describe().to_string(float_format='{:.5f}'.format) + '\n')
    train_recorder.write('Train predict stats: ' + y_train_pred.describe().to_string(float_format='{:.5f}'.format) + '\n')
    train_recorder.write('\nValidation Stats \n')
    train_recorder.write('Validation label stats: ' + y_valid.describe().to_string(float_format='{:.5f}'.format) + '\n')
    train_recorder.write('Validation predict stats: ' + y_valid_pred.describe().to_string(float_format='{:.5f}'.format) + '\n')

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

# file handler used to record training
if record:
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_recorder = open('data/error/%s_%s_params.txt' %(Model.__name__, time), 'w')
    train_recorder.write(Model.__name__ + '\n')
# split train_q4 into k folds, each time combine k-1 folds with train_q1_q3
# to train model and validate on the left out fold
mean_errors = []
models = []
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
# X_train_q4.to_csv('test_train_q4.csv')
for i, (train_index, validate_index) in enumerate(kf.split(X_train_q4)):
    X_train = pd.concat([X_train_q1_q3, X_train_q4.iloc[train_index]], ignore_index=True)
    y_train = pd.concat([y_train_q1_q3, y_train_q4.iloc[train_index]], ignore_index=True)
    # TODO(hzn): add training preprocessing, like remove outliers, resampling

    X_validate = X_train_q4.iloc[validate_index]
    y_validate = y_train_q4.iloc[validate_index]

    print('training...')
    model = Model()
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    print("fold train mean error: ", Evaluator.mean_error(train_pred, y_train))

    print('validating...')
    y_pred = model.predict(X_validate)
    if submit:
        models.append(model)
    # TODO(hzn): add output training records
    mae = Evaluator.mean_error(y_pred, y_validate)
    mean_errors.append(mae)
    print("fold validation mean error: ", mae)
    # Record this fold:
    if record:
        train_recorder.write('\nFold %d\n' %i)
        train_recorder.write('Parameters: %s\n' %model.get_params())
        feature_importances = model.get_features_importances()
        if feature_importances is not None:
            train_recorder.write('Feature importances:\n%s\n' %feature_importances)
            # feature_importances_map = list(zip(X_train.columns, feature_importances))
            # feature_importances_map.sort(key=lambda x: -x[1])
            # for fi in feature_importances_map:
            #     train_recorder.write('%s\n' %fi)
        record_train(train_recorder, y_train, train_pred, y_validate, y_pred)
    print("--------------------------------------------------------")

avg_cv_errors = np.mean(mean_errors)
if record:
    train_recorder.write("\nAverage cross validation mean error: %d\n" %avg_cv_errors)
    train_recorder.close()
print("average cross validation mean error", avg_cv_errors)

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
    # time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sample.to_csv(
        'data/submissions/Submission_%s.csv' %time, index=False, float_format='%.4f')
    print("submission generated.")
