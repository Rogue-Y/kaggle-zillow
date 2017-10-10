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
import train


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

    first_layer_csv_folder = 'data/ensemble/csv/validate'

    validation_csv_list = stacking_list['csv']
    configs = stacking_list['config']

    for config_dict, force_generate in configs:
        # Read config
        config_name = config_dict['name']
        validation2016_csv = '%s/%s2016.csv' %(first_layer_csv_folder, config_name)
        validation_all_csv = '%s/%s_all.csv' %(first_layer_csv_folder, config_name)
        # first_layer_pickle_path_test = '%s/%s_test' %(first_layer_pickle_folder, config_name)
        need_generate = (
            force_generate
            or global_force_generate
            or (not os.path.exists(validation2016_csv))
            or (not os.path.exists(validation_all_csv))
        )
            # if submit:
            #     if os.path.exists(first_layer_pickle_path_test):
            #         test_pred = pickle.load(open(first_layer_pickle_path_test, 'rb'))
            #     else:
            #         need_generate = True
        if need_generate:
            print('Generating first layer csv for config: %s ...' %config_name)
            train_config(config_dict, mode='stacking')
        validation_csv_list.append(config_name)

    # get validation targets:
    print('Loading validation target')
    transaction2016 = utils.load_transaction_data(2016)
    print(transaction2016.shape)
    transaction2017 = utils.load_transaction_data(2017)
    print(transaction2017.shape)

    _, df_validate2016, _, df_validate_all = train.get_train_validate_split(transaction2016, transaction2017)

    print(df_validate2016.shape)
    print(df_validate_all.shape)

    for config_name in validation_csv_list:
        validation2016_csv = '%s/%s2016.csv' %(first_layer_csv_folder, config_name)
        validation_all_csv = '%s/%s_all.csv' %(first_layer_csv_folder, config_name)
        validation2016 = pd.read_csv(validation2016_csv, parse_dates=['transactiondate'])
        validation_all = pd.read_csv(validation_all_csv, parse_dates=['transactiondate'])
        print(config_name)
        print(validation2016.shape)
        print(validation_all.shape)
        df_validate2016 = df_validate2016.merge(validation2016, 'left', ['parcelid', 'transactiondate'])
        df_validate_all = df_validate_all.merge(validation_all, 'left', ['parcelid', 'transactiondate'])

    df_validate2016.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    df_validate_all.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)

    print(df_validate2016.shape)
    print(df_validate_all.shape)

    first_layer2016, target2016 = utils.get_features_target(df_validate2016)
    first_layer_all, target_all = utils.get_features_target(df_validate_all)

    print('First layer generated.')
    print('First layer shape.')
    print(first_layer2016.shape, target2016.shape)
    print(first_layer_all.shape, target_all.shape)
    return first_layer2016, target2016, first_layer_all, target_all

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
        outliers_lw_pct = 0, outliers_up_pct = 100,
        config_dict={'name': 'fake_stacking_config'}, resale_offset=0.012):
    print(first_layer.shape, target.shape)
    assert len(first_layer) == len(target)

    print(first_layer_test.shape)

    ulimit = np.percentile(target.values, outliers_up_pct)
    llimit = np.percentile(target.values, outliers_lw_pct)
    mask = (target >= llimit) & (target <= ulimit)
    print(llimit, ulimit)
    first_layer = first_layer[mask]
    target = target[mask]
    print(first_layer.shape, target.shape)

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

        outliers_up_pct = config_dict['outliers_up_pct'] if 'outliers_up_pct' in config_dict else 100
        outliers_lw_pct = config_dict['outliers_lw_pct'] if 'outliers_lw_pct' in config_dict else 0

        # whether force generate all first layer
        global_force_generate = config_dict['global_force_generate'] if 'global_force_generate' in config_dict else False
        first_layer, first_layer_target, first_layer_test = get_first_layer(stacking_list, submit, global_force_generate)
        if submit:
            stacking_submit(first_layer, first_layer_target, first_layer_test, meta_model, outliers_lw_pct, outliers_up_pct, config_dict)
        else:
            stacking(first_layer, first_layer_target, meta_model, outliers_lw_pct, outliers_up_pct)

    t2 = time.time()
    print((t2 - t1) / 60)
