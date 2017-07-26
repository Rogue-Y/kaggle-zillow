# mimic feature_eng.utils.get_train_test_sets
import feature_eng.utils as utils
import feature_eng.data_clean as data_clean
import feature_eng.feature_eng as feature_eng

import evaluator

import json
import importlib # import module dynamically
import pandas as pd

# Read config file
config_file = open("config/steps.json", "r")
config = None
try:
    config = json.load(config_file)
finally:
    config_file.close()

steps = config['steps']
models = config['models']
predict = False
if 'predict' in config:
    predict = config['predict']

# Define global variables here
# prop_df, df= None, None

print('Loading data ...')
#TODO(hzn): Add a copy of train or prop dataframe if needed.
train, prop = utils.load_train_data()

print('Cleaning data and feature engineering...')
# Steps that must be done on the global data set.
step0 = steps[0]
# Steps that can be done on training set only, save time when we are not output
# predict data.
step1 = steps[1]
if predict:
    step0 = step0 + step1
for method in step0:
    module_name = method['module']
    module = globals()[module_name]
    method_name = method['method']
    method_to_call = getattr(module, method_name)

    params = method['params']
    args = params['args']
    kwargs = params['kwargs']

    args = list(map(lambda x: globals()[x], args))
    for key, value_name in kwargs.items():
        value = globals()[value_name]
        params[key] = value
    kwargs['df'] = prop

    prop = method_to_call(*args, **kwargs)

# Subset with transaction info
df = train.merge(prop, how='left', on='parcelid')

# Run some feature engineering jobs on trainning set only when not output
# prediction result.
if not predict:
    for method in step1:
        module_name = method['module']
        module = globals()[module_name]
        method_name = method['method']
        method_to_call = getattr(module, method_name)

        params = method['params']
        args = params['args']
        kwargs = params['kwargs']

        args = list(map(lambda x: globals()[x], args))
        for key, value_name in kwargs.items():
            value = globals()[value_name]
            params[key] = value
        kwargs['df'] = df

        df = method_to_call(*args, **kwargs)


print("Spliting data into training and testing...")
# transaction date is needed to split train and test(by ourselves) here.
train_df, test_df = utils.split_by_date(df)
train_df = data_clean.drop_training_only_column(train_df)
test_df = data_clean.drop_training_only_column(test_df)
# 82249 rows
X_train, y_train = utils.get_features_target(train_df)
# 8562 rows
X_test, y_test = utils.get_features_target(test_df)

# Evaluate
ev = evaluator.Evaluator()
ev.load_train_test((X_train, y_train, X_test, y_test))
for model in models:
    module_name = model['module']
    module = importlib.import_module(module_name)
    model_name = model['model']
    model_to_use = getattr(module, model_name)
    print('Using model', model_name)
    params = model['params']
    evparams = model['evparams']
    ev.fit(model_to_use(**params), **evparams)

# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

if predict:
    print("Predicting and writing results...")
    # load test set and output sample
    df_test, sample = utils.load_test_data()
    # organize test set
    df_test = df_test.merge(prop, on='parcelid', how='left')
    df_test = data_clean.drop_id_column(df_test)
    # Retrain predictor on the entire training set, then predict on test set
    df = data_clean.drop_training_only_column(df)
    X_df, y_df = utils.get_features_target(df)
    predictor = ev.predictors[0]['predictor']
    predictor.fit(X_df, y_df)
    p_test = predictor.predict(df_test)
    # Transform the distribution of the target if needed.
    transform_target = ev.predictors[0]['transform_target']
    if transform_target:
        p_test = ev.postprocess_target(p_test)
    for c in sample.columns[sample.columns != 'ParcelId']:
        sample[c] = p_test

    sample.to_csv('lgb_starter.csv', index=False, float_format='%.4f')
