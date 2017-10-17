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
def get_first_layer(stacking_list, submit=False, clean_na = False, global_force_generate=False):
    print('Generate first layer...')

    first_layer_csv_folder = 'data/ensemble/csv/validate'
    first_layer_test_csv_folder = 'data/ensemble/csv/test'


    validation_csv_list = stacking_list['csv']
    configs = stacking_list['config']

    for config_dict, force_generate in configs:
        # Read config
        config_name = config_dict['name']
        validation2016_csv = '%s/%s2016.csv' %(first_layer_csv_folder, config_name)
        validation_all_csv = '%s/%s_all.csv' %(first_layer_csv_folder, config_name)
        test_csv = '%s/%s.csv' %(first_layer_test_csv_folder, config_name)
        need_generate = (
            force_generate
            or global_force_generate
            or (not os.path.exists(validation2016_csv))
            or (not os.path.exists(validation_all_csv))
        )
        if submit:
            need_generate = need_generate or (not os.path.exists(test_csv))
        if need_generate:
            print('Generating first layer csv for config: %s ...' %config_name)
            train.train_config(config_dict, mode='stacking', submit=submit)
        validation_csv_list.append(config_name)

    # get validation targets:
    print('Loading validation target...')
    transaction2016 = utils.load_transaction_data(2016)
    transaction2017 = utils.load_transaction_data(2017)

    _, df_validate2016, _, df_validate_all = train.get_train_validate_split(transaction2016, transaction2017)

    print('Merge validation predictions...')
    if submit:
        df_test, _ = utils.load_test_data()
        test_first_layers = {}
        test_months = ['201610', '201611', '201612', '201710', '201711', '201712']
    for config_name in validation_csv_list:
        validation2016_csv = '%s/%s2016.csv' %(first_layer_csv_folder, config_name)
        validation_all_csv = '%s/%s_all.csv' %(first_layer_csv_folder, config_name)
        validation2016 = pd.read_csv(validation2016_csv, parse_dates=['transactiondate'])
        validation_all = pd.read_csv(validation_all_csv, parse_dates=['transactiondate'])
        # merge each validation predictions
        df_validate2016 = df_validate2016.merge(validation2016, 'left', ['parcelid', 'transactiondate'])
        df_validate_all = df_validate_all.merge(validation_all, 'left', ['parcelid', 'transactiondate'])

        if submit:
            test = pd.read_csv('%s/%s.csv'  %(first_layer_test_csv_folder, config_name))
            for month in test_months:
                if month not in test_first_layers:
                    test_first_layers[month] = df_test
                pred = test[['parcelid', month]].rename(columns={month: config_name})
                test_first_layers[month] = test_first_layers[month].merge(pred, 'left', 'parcelid')

    df_validate2016.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    df_validate_all.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)

    first_layer2016, target2016 = utils.get_features_target(df_validate2016)
    first_layer_all, target_all = utils.get_features_target(df_validate_all)

    if clean_na:
        first_layer2016.fillna(0.0167, inplace=True)
        first_layer_all.fillna(0.0167, inplace=True)

    if not submit:
        test_first_layers = None
    else:
        if clean_na:
            for key, item in test_first_layers.items():
                item.fillna(0.0167, inplace=True)

    print('First layer generated.')
    return first_layer2016, target2016, first_layer_all, target_all, test_first_layers

def stacking(first_layer, target, Meta_model, model_params, outliers_lw_pct = 0, outliers_up_pct = 100):
    print(first_layer.shape)
    print(target.shape)
    assert len(first_layer) == len(target)

    meta_model = Meta_model(model_params=model_params)


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

def stacking_wrapper(first_layer2016, target2016, first_layer_all, target_all, Meta_model, model_params, outliers_lw_pct = 0, outliers_up_pct = 100):
    loss2016 = stacking(first_layer2016, target2016, Meta_model, model_params, outliers_lw_pct, outliers_up_pct)
    loss_all = stacking(first_layer_all, target_all, Meta_model, model_params, outliers_lw_pct, outliers_up_pct)
    print('weighted mean average error %s' %((loss2016 + 2*loss_all)/3))

# def stacking_submit(first_layer, target, first_layer_test, meta_model,
#         outliers_lw_pct = 0, outliers_up_pct = 100):
#     print(first_layer.shape, target.shape)
#     assert len(first_layer) == len(target)
#
#     print(first_layer_test.shape)
#
#     ulimit = np.percentile(target.values, outliers_up_pct)
#     llimit = np.percentile(target.values, outliers_lw_pct)
#     mask = (target >= llimit) & (target <= ulimit)
#     print(llimit, ulimit)
#     first_layer = first_layer[mask]
#     target = target[mask]
#     print(first_layer.shape, target.shape)
#
#     print('second layer...')
#     print('training...')
#     meta_model.fit(first_layer, target)
#     train_pred = meta_model.predict(first_layer)
#     train_error = Evaluator.mean_error(train_pred, target)
#     print("fold train mean error: ", train_error)
#
#     print('predictions...')
#     return meta_model.predict(first_layer_test)

def train_meta_model(first_layer, target, Meta_model, model_params,
        outliers_lw_pct = 0, outliers_up_pct = 100):
    print(first_layer.shape, target.shape)
    assert len(first_layer) == len(target)

    meta_model = Meta_model(model_params=model_params)

    ulimit = np.percentile(target.values, outliers_up_pct)
    llimit = np.percentile(target.values, outliers_lw_pct)
    mask = (target >= llimit) & (target <= ulimit)
    print(llimit, ulimit)
    first_layer = first_layer[mask]
    target = target[mask]
    print(first_layer.shape, target.shape)

    print('training meta model...')
    meta_model.fit(first_layer, target)
    train_pred = meta_model.predict(first_layer)
    train_error = Evaluator.mean_error(train_pred, target)
    print("fold train mean error: ", train_error)
    return meta_model, train_error

def get_meta_model(config_file):
    config_dict = getattr(config, config_file)
    print('Config name: %s' %config_dict['name'])
    # Read stacking configuration
    # stacking list
    stacking_list = config_dict['stacking_list']
    # meta model
    Meta_model = config_dict['Meta_model']
    model_params = config_dict['model_params']

    outliers_up_pct = config_dict['outliers_up_pct'] if 'outliers_up_pct' in config_dict else 100
    outliers_lw_pct = config_dict['outliers_lw_pct'] if 'outliers_lw_pct' in config_dict else 0

    # clean_na
    clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False

    # whether force generate all first layer
    global_force_generate = config_dict['global_force_generate'] if 'global_force_generate' in config_dict else False
    first_layer2016, target2016, first_layer_all, target_all, _ = get_first_layer(stacking_list, False, clean_na, global_force_generate)
    meta_model2016, train_loss2016 = train_meta_model(first_layer2016, target2016, Meta_model, model_params, outliers_lw_pct, outliers_up_pct)
    meta_model2017, train_loss_all = train_meta_model(first_layer_all, target_all, Meta_model, model_params, outliers_lw_pct, outliers_up_pct)
    return meta_model2016, train_loss2016, meta_model2017, train_loss_all

def stacking_submit_wrapper(first_layer2016, target2016, first_layer_all, target_all,
        test_first_layers, Meta_model, model_params,
        outliers_lw_pct, outliers_up_pct, config_dict, resale_offset=0.012):
    # Get the cross validation error
    loss2016 = stacking(first_layer2016, target2016, Meta_model, model_params, outliers_lw_pct, outliers_up_pct)
    loss_all = stacking(first_layer_all, target_all, Meta_model, model_params, outliers_lw_pct, outliers_up_pct)
    avg_cv_errors = (loss2016 + 2 * loss_all) / 3

    months = ['10', '11', '12']

    # Generate submission
    print("loading submission data...")
    _, sample = utils.load_test_data()
    print(sample.shape)

    # 2016
    print('predicting 2016')
    meta_model2016, train_loss2016 = train_meta_model(first_layer2016, target2016, Meta_model, model_params, outliers_lw_pct, outliers_up_pct)
    for month in months:
        year_month = '2016' + month
        first_layer_test = test_first_layers[year_month].drop('parcelid', axis=1)
        sample[year_month] = meta_model2016.predict(first_layer_test)
        print(sample[year_month].describe())

    # make prediction
    print("Add resale 2016...")
    transactions2016 = utils.load_transaction_data(2016)
    # add resale
    sales2016 = transactions2016[['parcelid', 'logerror']].groupby('parcelid').mean()
    sample = sample.join(sales2016, on='ParcelId', how='left')
    for month in months:
        year_month = '2016' + month
        sample[year_month] = sample[year_month].where(
            sample['logerror'].isnull(),  sample[year_month] + resale_offset)
    sample.drop('logerror', axis=1, inplace=True)

    print('predicting 2017')
    meta_model2017, train_loss_all = train_meta_model(first_layer_all, target_all, Meta_model, model_params, outliers_lw_pct, outliers_up_pct)
    for month in months:
        year_month = '2017' + month
        first_layer_test = test_first_layers[year_month].drop('parcelid', axis=1)
        sample[year_month] = meta_model2017.predict(first_layer_test)
        print(sample[year_month].describe())

    # make prediction
    print("Add resale 2017...")
    transactions2017 = utils.load_transaction_data(2017)
    # add resale
    sales2017 = transactions2017[['parcelid', 'logerror']].groupby('parcelid').mean()
    sample = sample.join(sales2017, on='ParcelId', how='left')
    for month in months:
        year_month = '2017' + month
        sample[year_month] = sample[year_month].where(
            sample['logerror'].isnull(),  sample[year_month] + resale_offset)
    sample.drop('logerror', axis=1, inplace=True)

    # Sanity check, should not have nan
    print('Prediction sanity check')
    for col in sample:
        print(col, sample[col].isnull().sum())

    # generate submission
    print("generating submission...")
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    submission_folder = 'data/submissions'
    if not os.path.exists(submission_folder):
        os.makedirs(submission_folder)
    sample.to_csv(
        '%s/Submission_%s_%s.csv' %(submission_folder, time, config_dict['name']), index=False, float_format='%.6f')
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
        record.write('avg cv error:%s\n' %avg_cv_errors)
        record.write('train_error 2016:%s\n' %train_loss2016)
        record.write('train_error all:%s\n' %train_loss_all)
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

        outliers_up_pct = config_dict['outliers_up_pct'] if 'outliers_up_pct' in config_dict else 100
        outliers_lw_pct = config_dict['outliers_lw_pct'] if 'outliers_lw_pct' in config_dict else 0

        # clean_na
        clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False

        # whether force generate all first layer
        global_force_generate = config_dict['global_force_generate'] if 'global_force_generate' in config_dict else False
        first_layer2016, target2016, first_layer_all, target_all, test_first_layers = get_first_layer(stacking_list, submit, clean_na, global_force_generate)
        if submit:
            stacking_submit_wrapper(first_layer2016, target2016, first_layer_all, target_all, test_first_layers, Meta_model, model_params, outliers_lw_pct, outliers_up_pct, config_dict)
        else:
            stacking_wrapper(first_layer2016, target2016, first_layer_all, target_all, Meta_model, model_params, outliers_lw_pct, outliers_up_pct)

    t2 = time.time()
    print((t2 - t1) / 60)
