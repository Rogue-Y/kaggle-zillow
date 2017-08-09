# mimic feature_eng.utils.get_train_test_sets
import feature_eng.utils as utils
import feature_eng.data_clean as data_clean
import feature_eng.feature_eng as feature_eng
import evaluator

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import importlib # import module dynamically
import datetime
import gc

def load_train_data():
    print('Loading data ...')
    #TODO(hzn): Add a copy of train or prop dataframe if needed.
    return utils.load_train_data()

def load_config():
    # Read config file
    print('Loading config ...')
    # Note that we should not alter the config object (i.e. it should be read-only)
    # for the auto change detection in main.py to work properly.
    return utils.load_config()

def process_data(train, prop, config):
    steps = config['steps']
    # preprocessing for training data
    training_preprocess = config['training_preprocess']
    predict = config['predict'] if 'predict' in config else False

    # Define global variables here
    # prop_df, df= None, None

    print('Cleaning data and feature engineering...')
    # Steps that must be done on the global data set.
    step0 = steps[0]
    # Steps that can be done on training set only, save time when we are not output
    # predict data.
    step1 = steps[1]
    if predict:
        step0 = step0 + step1
    for method in step0:
        active = method['active'] if 'active' in method else True
        if not active:
            continue
        module_name = method['module']
        module = globals()[module_name]
        method_name = method['method']
        method_to_call = getattr(module, method_name)

        params = method['params']
        args = params['args']
        kwargs = params['kwargs']

        # args = list(map(lambda x: globals()[x], args))
        # for key, value_name in kwargs.items():
        #     value = globals()[value_name]
        #     params[key] = value

        prop = method_to_call(*args, **kwargs, df=prop)

    # Subset with transaction info
    df = train.merge(prop, how='left', on='parcelid')

    # Run some feature engineering jobs on trainning set only when not output
    # prediction result.
    if not predict:
        for method in step1:
            active = method['active'] if 'active' in method else True
            if not active:
                continue
            module_name = method['module']
            module = globals()[module_name]
            method_name = method['method']
            method_to_call = getattr(module, method_name)

            params = method['params']
            args = params['args']
            kwargs = params['kwargs']

            # args = list(map(lambda x: globals()[x], args))
            # for key, value_name in kwargs.items():
            #     value = globals()[value_name]
            #     params[key] = value

            df = method_to_call(*args, **kwargs, df=df)

    print("The shape of the dataframe: {0}\n".format(df.shape))
    # for col in df.columns:
    #     print(col)

    print("Spliting data into training and testing...")
    # transaction date is needed to split train and test(by ourselves) here.
    train_df, test_df = utils.split_by_date(df)

    # preprocess training data
    for method in training_preprocess:
        active = method['active'] if 'active' in method else True
        if not active:
            continue
        module_name = method['module']
        module = globals()[module_name]
        method_name = method['method']
        method_to_call = getattr(module, method_name)

        params = method['params']
        args = params['args']
        kwargs = params['kwargs']

        # args = list(map(lambda x: globals()[x], args))
        # for key, value_name in kwargs.items():
        #     value = globals()[value_name]
        #     params[key] = value

        train_df = method_to_call(*args, **kwargs, df=train_df)
        test_df = method_to_call(*args, **kwargs, df=test_df)

    # Drop columns that are only available in training data
    train_df = data_clean.drop_training_only_column(train_df)
    test_df = data_clean.drop_training_only_column(test_df)
    # 82249 rows
    X_train, y_train = utils.get_features_target(train_df)
    # 8562 rows
    X_test, y_test = utils.get_features_target(test_df)

    return (X_train, y_train, X_test, y_test, prop)

def train(X_train, y_train, X_test, y_test, prop, config):
    print("Training models...")
    # Read relevant config
    models = config['models']
    develop_mode = config['develop_mode'] if 'develop_mode' in config else True
    predict = config['predict'] if 'predict' in config else False
    # Evaluate
    ev = evaluator.Evaluator()
    ev.load_train_test((X_train, y_train, X_test, y_test))
    for model in models:
        # skip models marked as not active, active is default to True
        active = model['active'] if 'active' in model else True
        if not active:
            continue
        module_name = model['module']
        module = importlib.import_module(module_name)
        # Reload the module to make sure include the changes. This is necessary
        # when developing as importlib.import_module will not reload module when
        # it is already loaded.
        if develop_mode:
            importlib.reload(module)
        model_name = model['model']
        model_to_use = getattr(module, model_name)
        print('Using model', model_name)
        evparams = model['evparams']
        params = model['params'] # should be the param space when grid search
        predictor = model_to_use(**params)
        grid_search = model['grid_search'] if 'grid_search' in model else False
        if grid_search:
            grid_search_params = model['grid_search_params'] # params of GridSearchCV
            param_space = model['param_space']
            ev.grid_search(predictor, param_space, grid_search_params,
                **evparams, save_result=predict)
        else:
            predictor = model_to_use(**params)
            ev.fit(predictor, **evparams, predictor_params=params,
            save_result=predict)
        # print some attributes of the model
        if "attributes" in model:
            for attr in model["attributes"]:
                attribute = getattr(predictor, attr)
                print(attribute)

    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    if predict:
        print("Predicting and writing results...")
        # load test set and output sample
        df_test, sample = utils.load_test_data()
        # organize test set
        df_test = df_test.merge(prop, on='parcelid', how='left')
        df_test = data_clean.drop_id_column(df_test)
        # Retrain predictor on the entire training set, then predict on test set
        X_df, y_df = pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
        # At this point, X_train, X_test, y_train, y_test is still stored in ev

        # Ensembling
        result = np.zeros(df_test.shape[0])
        validate_result = None
        total_weight = 0
        # stacking_df = pd.DataFrame()
        # result_df = pd.DataFrame()
        for predictor_dict in ev.predictors:
            predictor = predictor_dict['predictor']
            grid_search = predictor_dict['grid_search']
            if grid_search:
                predictor = predictor.best_estimator_
            predictor.fit(X_df, y_df)
            # p_train = predictor.predict(X_df)
            # TODO(hzn): investigate using the average of k-fold to replace this
            # re-train
            p_test = predictor.predict(df_test)
            # Transform the distribution of the target if needed.
            transform_target = predictor_dict['transform_target']
            if transform_target:
                # p_train = ev.postprocess_target(p_train)
                p_test = ev.postprocess_target(p_test)
            # # Add the predictions to dataframes for stacking
            # model_name = predictor_dict['model_name']
            # stacking_df[model_name] = p_train
            # result_df[model_name] = p_test
            weight = predictor_dict['weight']
            total_weight += weight
            result = np.add(result, weight * p_test)
            # Ensemble validate result to get a sense of how the ensembling will
            # work
            p_validate = predictor_dict['y_test_predict']
            if validate_result is None:
                validate_result = weight * p_validate
            else:
                validate_result = np.add(validate_result, weight * p_validate)
        if total_weight > 0:
            result = result / total_weight
            validate_result = validate_result / total_weight
        print("Ensembling validate:", ev.mean_error(validate_result, ev.y_test))
        # stacking_predictor = LinearRegression()
        # stacking_predictor.fit(stacking_df, y_df)
        # result = stacking_predictor.predict(result_df)

        for c in sample.columns[sample.columns != 'ParcelId']:
            sample[c] = result
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sample.to_csv(
            'data/Submission_%s.csv' %time, index=False, float_format='%.4f')
        print("Submission generated.")

    # Return useful information for notebook analysis use
    # return df, ev

if __name__ == "__main__":
    data = load_train_data()
    config = load_config()
    training_data = process_data(*data, config)
    train(*training_data, config)
