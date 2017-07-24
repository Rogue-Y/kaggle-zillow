# mimic feature_eng.utils.get_train_test_sets
import feature_eng.utils as utils
import feature_eng.data_clean as data_clean
import feature_eng.feature_eng as feature_eng

import evaluator

import json
import importlib # import module dynamically

# Read config file
steps_config_file = open("config/steps.json", "r")
config = None
try:
    config = json.load(steps_config_file)
finally:
    steps_config_file.close()

steps = config['steps']
models = config['models']

# Global variables
prop_df, df= None, None

print('Loading data ...')
train, prop = utils.load_train_data()

print('Cleaning data and feature engineering...')
step0 = steps[0]
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

    prop_df = method_to_call(*args, **kwargs)

# Subset with transaction info
df = train.merge(prop_df, how='left', on='parcelid')
step1 = steps[1]
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

    df = method_to_call(*args, **kwargs)

print("Spliting data into training and testing...")
train_df, test_df = utils.split_by_date(df)
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
