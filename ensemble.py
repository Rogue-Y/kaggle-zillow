# Ensembling prediction result
# TODO(hzn):
#   *1. how to record ensembling experiements result
#   *2. how to envaluate ensembling in cv rather than submit the result, predictions
#       actually contains the training houses
#   *3. following the above one, how to automatically tune ensembling
#   *3. adjust for resale first or ensembling first (mathematically the same,
#       but the adjust offset could be different for different ensembling, also
#       the adjustment may affect validation, validation should be made with
#       pre-adjusted predictions)

import datetime
import gc
import os
import pickle
import time
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import config
from evaluator import Evaluator
from features import utils
from train import prepare_features, prepare_training_data, train_stacking


### Average ensembling ###

# tuples of format (pickle_path, weight)
prediction_list = [
    # ('XGBoost_latest_pickle', 7),
    ('Lightgbm_latest_pickle', 3)
]

def ensemble(prediction_list=prediction_list):
    print('Ensembling...')
    print('loading sample submission...')
    df_test, sample = utils.load_test_data()
    # ensembling
    print('ensembling...')
    # initialize the prediction with bias, currently 0
    pred_sum = np.zeros(sample.shape[0])
    total_weight = 0
    for pickle_path, weight in prediction_list:
        pred = pd.read_pickle('data/predictions/%s' %pickle_path)
        pred = weight * pred.as_matrix()
        total_weight += weight
        pred_sum += pred
    ensembled_pred = pred_sum / total_weight

    # generate submission
    print("generating submission...")
    for c in sample.columns[sample.columns != 'ParcelId']:
        # sample[c] = avg_pred
        sample[c] = ensembled_pred
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sample.to_csv(
        'data/submissions/Submission_%s.csv' %time, index=False, float_format='%.4f')



### Stacking ###

def get_first_layer(stacking_list, submit=False, global_force_generate=False):
    print('Generate first layer...')

    first_layer_preds = []
    first_layer_preds_test = []

    for config_dict, force_generate in stacking_list:
        # Read config
        config_name = config_dict['name']
        first_layer_pickle_folder = 'data/ensemble/first_layer'
        first_layer_pickle_path_validation = '%s/%s_validation' %(first_layer_pickle_folder, config_name)
        first_layer_pickle_path_test = '%s/%s_test' %(first_layer_pickle_folder, config_name)
        first_layer_pickle_path_target = '%s/target' %first_layer_pickle_folder
        need_generate = force_generate or global_force_generate
        if not need_generate:
            if os.path.exists(first_layer_pickle_path_validation):
                validation_pred = pickle.load(open(first_layer_pickle_path_validation, 'rb'))
            else:
                need_generate = True
            if os.path.exists(first_layer_pickle_path_target):
                validation_target = pickle.load(open(first_layer_pickle_path_target, 'rb'))
            else:
                need_generate = True
            if submit:
                if os.path.exists(first_layer_pickle_path_test):
                    test_pred = pickle.load(open(first_layer_pickle_path_test, 'rb'))
                else:
                    need_generate = True
        if need_generate:
            print('Generating first layer for config: %s ...' %config_name)
            # Mandatory configurations:
            # Feature list
            feature_list = config_dict['feature_list']
            # Model
            Model = config_dict['Model']

            # clean_na
            clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False

            prop = prepare_features(feature_list, clean_na)
            train_df, transactions = prepare_training_data(prop)
            del transactions; gc.collect()
            if not submit:
                prop = None
                gc.collect()

            validation_pred, validation_target, test_pred = train_stacking(
                train_df, Model=Model,
                submit=submit, config_name=config_name, prop=prop,
                **config_dict['stacking_params'])

            if not os.path.exists(first_layer_pickle_folder):
                os.makedirs(first_layer_pickle_folder)
            pickle.dump(validation_pred, open(first_layer_pickle_path_validation, 'wb'))
            pickle.dump(validation_target, open(first_layer_pickle_path_target, 'wb'))
            if submit:
                pickle.dump(test_pred, open(first_layer_pickle_path_test, 'wb'))

        first_layer_preds.append(validation_pred)
        if submit:
            first_layer_preds_test.append(test_pred)

    # assemble first layer result
    first_layer = pd.concat(first_layer_preds, axis=1)
    if submit:
        first_layer_test = pd.concat(first_layer_preds_test, axis=1)
    else:
        first_layer_test = None

    print('First layer generated.')
    return first_layer, validation_target, first_layer_test

def stacking(first_layer, target, meta_model, outliers_lw_pct = 0, outliers_up_pct = 100):
    print(first_layer.shape)
    print(target.shape)
    assert len(first_layer) == len(target)

    print('second layer...')
    mean_errors = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    # same k fold split as above
    # (this is guaranteed as long as the random seeds and the number of rows does not change,
    # therefore we only need to make sure the indexes of the dataframes are always aligned)
    for train_index, validate_index in kf.split(first_layer):
        X_train, y_train = first_layer.loc[train_index], target.loc[train_index]
        X_validate, y_validate = first_layer.loc[validate_index], target.loc[validate_index]

        print(X_train.shape, y_train.shape)
        ulimit = np.percentile(y_train.values, outliers_up_pct)
        llimit = np.percentile(y_train.values, outliers_lw_pct)
        mask = (y_train >= llimit) & (y_train <= ulimit)
        print(llimit, ulimit)
        X_train = X_train[mask]
        y_train = y_train[mask]
        print(X_train.shape, y_train.shape)

        print('training...')
        meta_model.fit(X_train, y_train)
        train_pred = meta_model.predict(X_train)
        print("fold train mean error: ", Evaluator.mean_error(train_pred, y_train))

        print('validating...')
        y_pred = meta_model.predict(X_validate)
        # TODO(hzn): add output training records
        mae = Evaluator.mean_error(y_pred, y_validate)
        mean_errors.append(mae)
        print("fold validation mean error: ", mae)

    avg_cv_errors = np.mean(mean_errors)
    print("average cross validation mean error", avg_cv_errors)
    return avg_cv_errors

def stacking_submit(first_layer, target, first_layer_test, meta_model,
        config_dict={'name': 'fake_stacking_config'}, resale_offset=0.012):
    print(first_layer.shape)
    print(target.shape)
    assert len(first_layer) == len(target)

    print(first_layer_test.shape)

    print('second layer...')
    print('training...')
    meta_model.fit(first_layer, target)
    train_pred = meta_model.predict(first_layer)
    train_error = Evaluator.mean_error(train_pred, target)
    print("fold train mean error: ", train_error)

    print('predictions...')
    pred = meta_model.predict(first_layer_test)

    # Generate submission
    print("loading submission data...")
    predict_df, sample = utils.load_test_data()
    print(predict_df.shape)
    print(sample.shape)

    # make prediction
    print("make prediction...")
    transactions = utils.load_transaction_data()
    # add resale
    sales = transactions[['parcelid', 'logerror']].groupby('parcelid').mean()
    predict_df = predict_df.join(sales, on='parcelid', how='left')
    predict_df['predict'] = pred
    predict = predict_df['predict'].where(
        predict_df['logerror'].isnull(), predict_df['predict'] + resale_offset)
    # Sanity check, should not have nan
    print(predict.isnull().sum())

    # generate submission
    print("generating submission...")
    for c in sample.columns[sample.columns != 'ParcelId']:
        # sample[c] = avg_pred
        sample[c] = predict.as_matrix()
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    submission_folder = 'data/submissions'
    if not os.path.exists(submission_folder):
        os.makedirs(submission_folder)
    sample.to_csv(
        '%s/Submission_%s_%s.csv' %(submission_folder, time, config_dict['name']), index=False, float_format='%.4f')
    print("Prediction made.")

    # Record the generated submissions for future comparison.
    exp_record_folder = 'data/experiments'
    if not os.path.exists(exp_record_folder):
        os.makedirs(exp_record_folder)
    with open('%s/%s.txt' %(exp_record_folder, 'stacking_experiments'), 'a') as record:
         # Time is the same across submission csv, pickle and record for easy search
        record.write('\n\nTime: %s\n' %time)
        record.write('Config: %s\n' %config_dict)
        # TODO: add cross validation step in stacking submission.
        record.write('train_error:%s\n' %train_error)
        record.write('leaderboard:________________PLEASE FILL____________________\n')


# Main method
if __name__ == '__main__':
    t1 = time.time()
    # Get configuration
    # parser to parse cmd line option
    parser = OptionParser()
    # ensemble is true by default or when -e flag is present
    parser.add_option('-e', '--ensemble', action='store_true', dest='ensemble', default=True)
    # ensemble is set to false when -s flag is present
    parser.add_option('-t', '--stacking', action='store_false', dest='ensemble')
    # configuration dictionary for stacking
    parser.add_option('-c', '--config', action='store', type='string', dest='config_file')
    # if generate submisstion
    parser.add_option('-s', '--submit', action='store_true', dest='submit', default=False)
    # parse cmd line arguments
    (options, args) = parser.parse_args()

    if options.ensemble:
        ensemble()
    else:
        # If generate submission
        submit = options.submit

        config_file = options.config_file
        config_dict = getattr(config, config_file)
        print('Config name: %s' %config_dict['name'])
        # Read stacking configuration
        # stacking list
        stacking_list = config_dict['stacking_list']
        # meta model
        Meta_model = config_dict['Meta_model']
        model_params = config_dict['model_params']
        meta_model = Meta_model(model_params=model_params)
        # whether force generate all first layer
        global_force_generate = config_dict['global_force_generate'] if 'global_force_generate' in config_dict else False
        first_layer, first_layer_target, first_layer_test = get_first_layer(stacking_list, submit, global_force_generate)
        if submit:
            stacking_submit(first_layer, first_layer_target, first_layer_test, meta_model, config_dict)
        else:
            stacking(first_layer, first_layer_target, meta_model)

    t2 = time.time()
    print((t2 - t1) / 60)
