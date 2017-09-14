# Train model and cross validation
# TODO(hzn):
#   1. data clean step, like drop low ratio columns, fill nan in the original prop;
#   2. training data proprocessing
#   3. model wrapper, model parameter tuning
#   4. notebook api
#   5. output training records
#   6. write code to automatically add feature and see how it works: not working
#   7. move configuration to a standalone file, and read it from cmd line
#  *8. add ensembler, how to better preseve predict result and their timestamp,
#       so that: 1. it's easy for ensembler to identify them; 2, they can be
#               associated with other record like training record.

import datetime
import gc
import os
import sys
import time
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

import config
from evaluator import Evaluator
from features import utils
from features import feature_combine, feature_clean
from features import data_clean

# Helper functions:
def record_train(train_recorder, y_train, y_train_pred, y_valid, y_valid_pred):
    mean_train_error = Evaluator.mean_error(y_train, y_train_pred)
    mean_valid_error = Evaluator.mean_error(y_valid, y_valid_pred)
    y_train = pd.Series(y_train)
    y_train_pred = pd.Series(y_train_pred)
    y_valid = pd.Series(y_valid)
    y_valid_pred = pd.Series(y_valid_pred)
    train_recorder.write('Train error: ' + str(mean_train_error) + '\n')
    train_recorder.write('Validation error: ' + str(mean_valid_error) + '\n')
    train_recorder.write('\nTrain Stats \n')
    train_recorder.write('Train label stats: ' + y_train.describe().to_string(float_format='{:.5f}'.format) + '\n')
    train_recorder.write('Train predict stats: ' + y_train_pred.describe().to_string(float_format='{:.5f}'.format) + '\n')
    train_recorder.write('\nValidation Stats \n')
    train_recorder.write('Validation label stats: ' + y_valid.describe().to_string(float_format='{:.5f}'.format) + '\n')
    train_recorder.write('Validation predict stats: ' + y_valid_pred.describe().to_string(float_format='{:.5f}'.format) + '\n')

def prepare_features(feature_list = [], force_prepare=True, save_pickle=False):
    feature_eng_pickle = 'data/feature_eng_pickle'
    if not force_prepare and not save_pickle and os.path.exists(feature_eng_pickle):
        prop = pd.read_pickle(feature_eng_pickle)
    else:
        prop = utils.load_properties_data()
        # use minimized version of properties data when memory is a concern
        # prop = utils.load_properties_data_minimize()
        # feature engineering
        print('Feature engineering')
        prop = feature_combine.original_feature_clean(prop, feature_clean, False, 'features/feature_pickles/')
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
        if save_pickle:
            # Only save pickle when necessary, as this takes some time
            prop.to_pickle(feature_eng_pickle)
    print(prop.shape)
    return prop

def prepare_training_data(prop, clean_na = False):
    # Process:
    # load training data
    print('Load training data...')
    transactions = utils.load_transaction_data()

    # merge transaction and prop data
    train_df = transactions.merge(prop, how='left', on='parcelid')
    # df.to_csv('test_df.csv')
    # del train_df; gc.collect()

    if clean_na:
        #TODO: Configurize processing nan for different columns
        train_df = data_clean.clean_strange_value(train_df)

    return train_df, transactions

def train(train_df, Model, model_params = None, FOLDS = 5, record=False,
    outliers_up_pct = 99, outliers_lw_pct = 1,
    submit=False, prop = None, transactions = None, # if submit is true, than must provide transactions and prop
    resale_offset = 0.012, pca_components=-1):
    # Optional dimension reduction.
    if pca_components > 0:
        print('PCA...')
        pca = PCA(n_components=pca_components, copy=False)
        feature_df, non_feature_df = utils.get_dimension_reduction_df(train_df)
        # Note that the features pca produces is some combination of the original features, not retain/discard some columns
        feature_df = pca.fit_transform(feature_df)
        train_df = pd.concat([pd.DataFrame(feature_df), non_feature_df], axis=1, copy=False)

    # split by date
    train_q1_q3, train_q4 = utils.split_by_date(train_df)
    # train_q4.to_csv('test_train_q4.csv')
    del train_df; gc.collect()

    train_q1_q3 = data_clean.drop_training_only_column(train_q1_q3)
    train_q4 = data_clean.drop_training_only_column(train_q4)
    X_train_q1_q3, y_train_q1_q3 = utils.get_features_target(train_q1_q3)
    X_train_q4, y_train_q4 = utils.get_features_target(train_q4)
    del train_q1_q3; del train_q4; gc.collect()

    # file handler used to record training
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if record:
        train_recorder = open('data/error/%s_%s_params.txt' %(Model.__name__, time), 'w')
        train_recorder.write(Model.__name__ + '\n')
    # If need to generate submission, prepare prop df for testing
    if submit:
        df_test = data_clean.drop_id_column(prop)
    else:
        del prop; gc.collect()
    # split train_q4 into k folds, each time combine k-1 folds with train_q1_q3
    # to train model and validate on the left out fold
    mean_errors = []
    model_preds = []
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    # X_train_q4.to_csv('test_train_q4.csv')
    for i, (train_index, validate_index) in enumerate(kf.split(X_train_q4)):
        X_train = pd.concat([X_train_q1_q3, X_train_q4.iloc[train_index]], ignore_index=True)
        y_train = pd.concat([y_train_q1_q3, y_train_q4.iloc[train_index]], ignore_index=True)
        # TODO(hzn): add training preprocessing, like remove outliers, resampling

        X_validate = X_train_q4.iloc[validate_index]
        y_validate = y_train_q4.iloc[validate_index]

        # try remove outliers
        print(X_train.shape, y_train.shape)
        ulimit = np.percentile(y_train.values, outliers_up_pct)
        llimit = np.percentile(y_train.values, outliers_lw_pct)
        mask = (y_train >= llimit) & (y_train <= ulimit)
        print(llimit, ulimit)
        X_train = X_train[mask]
        y_train = y_train[mask]
        print(X_train.shape, y_train.shape)

        print('training...')
        model = Model(model_params = model_params)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        print("fold train mean error: ", Evaluator.mean_error(train_pred, y_train))

        print('validating...')
        y_pred = model.predict(X_validate)
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
        # Predict on testing data if needed to generate submission
        if submit:
            print('predicting on testing data...')
            test_preds = []
            # Split testing dataframe 
            for df_test_split in np.array_split(df_test, 30):
                test_preds.append(model.predict(df_test_split))
            model_preds.append(np.concatenate(test_preds))
        print("--------------------------------------------------------")

    avg_cv_errors = np.mean(mean_errors)
    if record:
        train_recorder.write("\nAverage cross validation mean error: %f\n" %avg_cv_errors)
        train_recorder.close()
    print("average cross validation mean error", avg_cv_errors)

    if submit:
        print("loading submission data...")
        df_test, sample = utils.load_test_data()
        print(df_test.shape)
        print(sample.shape)
        # keep a copy of to generate resale
        # predict_df = df_test.copy()
        predict_df = df_test
        # organize test set
        # df_test = df_test.merge(prop, on='parcelid', how='left')
        # df_test = data_clean.drop_id_column(df_test)

        # make prediction
        print("make prediction...")
        # model_preds = list(map(lambda model: model.predict(df_test), models))
        avg_pred = np.mean(model_preds, axis=0)
        print(len(avg_pred))

        # add resale
        sales = transactions[['parcelid', 'logerror']].groupby('parcelid').mean()
        predict_df = predict_df.join(sales, on='parcelid', how='left')
        predict_df['predict'] = avg_pred
        # predict = predict_df['predict'].where(predict_df['logerror'].isnull(), predict_df['logerror'])
        predict = predict_df['predict'].where(
            predict_df['logerror'].isnull(), predict_df['predict'] + resale_offset)

        # Save prediction(a Series object) to a pickle for ensembling use
        # Save one in history
        predict.to_pickle('data/predictions/history/%s_%s_pickle' %(time, Model.__name__))
        # Update the most recent pickle for this model
        predict.to_pickle('data/predictions/%s_latest_pickle' %Model.__name__)
        print("Prediction made.")

        # # generate submission
        # print("generating submission...")
        # for c in sample.columns[sample.columns != 'ParcelId']:
        #     # sample[c] = avg_pred
        #     sample[c] = predict.as_matrix()
        # # time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # sample.to_csv(
        #     'data/submissions/Submission_%s.csv' %time, index=False, float_format='%.4f')

    return avg_cv_errors


if __name__ == '__main__':
    t1 = time.time()
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
    # print configuration for confirmation
    for key, value in config_dict.items():
        if key == 'feature_list':
            print('%s: %s' %(key, len(value)))
        else:
            print('%s: %s' %(key, value))

    # Mandatory configurations:
    # Feature list
    feature_list = config_dict['feature_list']
    # model
    Model = config_dict['model']

    # Optional configurations:
    # model params:
    model_params = config_dict['model_params'] if 'model_params' in config_dict else None
    # folds number of K-Fold
    FOLDS = config_dict['folds'] if 'folds' in config_dict else 5
    # if record training
    record = config_dict['record'] if 'record' in config_dict else False
    # if generate submission or not
    submit = config_dict['submit'] if 'submit' in config_dict else False
    # outliers removal upper and lower percentile
    outliers_up_pct = config_dict['outliers_up_pct'] if 'outliers_up_pct' in config_dict else 99
    outliers_lw_pct = config_dict['outliers_lw_pct'] if 'outliers_lw_pct' in config_dict else 1
    # resale offset
    resale_offset = config_dict['resale_offset'] if 'resale_offset' in config_dict else 0.012
    # clean_na
    clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False
    # PCA component
    pca_components = config_dict['pca_components'] if 'pca_components' in config_dict else -1
    # PCA cannot deal with infinite numbers
    if pca_components > 0:
        clean_na = True


    prop = prepare_features(feature_list)

    train_df, transactions = prepare_training_data(prop, clean_na)
    if submit:
        cv_error = train(train_df,
            prop=prop, transactions=transactions, Model=Model, model_params=model_params, FOLDS = FOLDS,
            record=record, submit=True, outliers_up_pct=outliers_up_pct,
            outliers_lw_pct=outliers_lw_pct, resale_offset=resale_offset, pca_components=pca_components)
        folder = 'data/experiments'
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open('%s/%s.txt' %(folder, 'experiments'), 'a') as record:
            exp_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record.write('\n\nTime: %s\n' %exp_time)
            record.write('Config: %s\n' %config_dict)
            record.write('cv_error:%s\n' %cv_error)
            record.write('leaderboard:________________PLEASE FILL____________________\n')
    else:
        del transactions; del prop; gc.collect()
        train(train_df,
            Model=Model, model_params=model_params, FOLDS = FOLDS,
            record=record, submit=False, outliers_up_pct=outliers_up_pct,
            outliers_lw_pct=outliers_lw_pct, resale_offset=resale_offset, pca_components=pca_components)

    t2 = time.time()
    print((t2 - t1) / 60)
