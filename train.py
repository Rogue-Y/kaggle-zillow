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
from sklearn.preprocessing import RobustScaler

import config
from evaluator import Evaluator
from features import utils
from features import feature_combine, feature_clean
from features import data_clean

# TODO: For linear predictor, need to:
# 1. Add one hot encoding for categorical(including some geo, lat lon block) columns
# 2. Go through features one by one to see which should be include/exclude/modified, create a feature list for linear models

# columns that need to be scaled for certain predictors
# TODO: think of a better way to specify this
SCALING_COLUMNS = [
    'basementsqft',
    'bathroomcnt',
    'bedroomcnt',
    'buildingqualitytypeid',
    'calculatedbathnbr',
    'finishedfloor1squarefeet',
    'calculatedfinishedsquarefeet',
    'finishedsquarefeet12',
    'finishedsquarefeet13',
    'finishedsquarefeet15',
    'finishedsquarefeet50',
    'finishedsquarefeet6',
    'fullbathcnt',
    'garagecarcnt',
    'garagetotalsqft',
    'latitude',
    'longitude',
    'lotsizesquarefeet',
    'poolsizesum',
    'roomcnt',
    'threequarterbathnbr',
    'unitcnt',
    'yardbuildingsqft17',
    'yardbuildingsqft26',
    'yearbuilt',
    'numberofstories',
    'fireplaceflag',
    'structuretaxvaluedollarcnt',
    'taxvaluedollarcnt',
    'landtaxvaluedollarcnt',
    'taxamount',
    'taxdelinquencyyear',
    'missing_value_count',
    'average_bathroom_size',
    'average_bedroom_size',
    'average_room_size',
    'building_age',
    'error_rate_bathroom',
    'error_rate_calculated_finished_living_sqft',
    'error_rate_count_bathroom',
    'error_rate_first_floor_living_sqft',
    'extra_rooms',
    'extra_space',
    'city_count',
    'bathroomcnt_city_mean',
    'bathroomcnt_city_std',
    'bathroomcnt_city_mean_ratio',
    'bathroomcnt_city_std_ratio',
    'bedroomcnt_city_mean',
    'bedroomcnt_city_std',
    'bedroomcnt_city_mean_ratio',
    'bedroomcnt_city_std_ratio',
    'buildingqualitytypeid_city_mean',
    'buildingqualitytypeid_city_std',
    'buildingqualitytypeid_city_mean_ratio',
    'buildingqualitytypeid_city_std_ratio',
    'calculatedfinishedsquarefeet_city_mean',
    'calculatedfinishedsquarefeet_city_std',
    'calculatedfinishedsquarefeet_city_mean_ratio',
    'calculatedfinishedsquarefeet_city_std_ratio',
    'fullbathcnt_city_mean',
    'fullbathcnt_city_std',
    'fullbathcnt_city_mean_ratio',
    'fullbathcnt_city_std_ratio',
    'garagecarcnt_city_mean',
    'garagecarcnt_city_std',
    'garagecarcnt_city_mean_ratio',
    'garagecarcnt_city_std_ratio',
    'unitcnt_city_mean',
    'unitcnt_city_std',
    'unitcnt_city_mean_ratio',
    'unitcnt_city_std_ratio',
    'yearbuilt_city_mean',
    'yearbuilt_city_std',
    'yearbuilt_city_mean_ratio',
    'yearbuilt_city_std_ratio',
    'structuretaxvaluedollarcnt_city_mean',
    'structuretaxvaluedollarcnt_city_std',
    'structuretaxvaluedollarcnt_city_mean_ratio',
    'structuretaxvaluedollarcnt_city_std_ratio',
    'taxamount_city_mean',
    'taxamount_city_std',
    'taxamount_city_mean_ratio',
    'taxamount_city_std_ratio',
    'taxvaluedollarcnt_city_mean',
    'taxvaluedollarcnt_city_std',
    'taxvaluedollarcnt_city_mean_ratio',
    'taxvaluedollarcnt_city_std_ratio',
    'county_count',
    'bathroomcnt_county_mean',
    'bathroomcnt_county_std',
    'bathroomcnt_county_mean_ratio',
    'bathroomcnt_county_std_ratio',
    'bedroomcnt_county_mean',
    'bedroomcnt_county_std',
    'bedroomcnt_county_mean_ratio',
    'bedroomcnt_county_std_ratio',
    'buildingqualitytypeid_county_mean',
    'buildingqualitytypeid_county_std',
    'buildingqualitytypeid_county_mean_ratio',
    'buildingqualitytypeid_county_std_ratio',
    'calculatedfinishedsquarefeet_county_mean',
    'calculatedfinishedsquarefeet_county_std',
    'calculatedfinishedsquarefeet_county_mean_ratio',
    'calculatedfinishedsquarefeet_county_std_ratio',
    'fullbathcnt_county_mean',
    'fullbathcnt_county_std',
    'fullbathcnt_county_mean_ratio',
    'fullbathcnt_county_std_ratio',
    'garagecarcnt_county_mean',
    'garagecarcnt_county_std',
    'garagecarcnt_county_mean_ratio',
    'garagecarcnt_county_std_ratio',
    'unitcnt_county_mean',
    'unitcnt_county_std',
    'unitcnt_county_mean_ratio',
    'unitcnt_county_std_ratio',
    'yearbuilt_county_mean',
    'yearbuilt_county_std',
    'yearbuilt_county_mean_ratio',
    'yearbuilt_county_std_ratio',
    'structuretaxvaluedollarcnt_county_mean',
    'structuretaxvaluedollarcnt_county_std',
    'structuretaxvaluedollarcnt_county_mean_ratio',
    'structuretaxvaluedollarcnt_county_std_ratio',
    'taxamount_county_mean',
    'taxamount_county_std',
    'taxamount_county_mean_ratio',
    'taxamount_county_std_ratio',
    'taxvaluedollarcnt_county_mean',
    'taxvaluedollarcnt_county_std',
    'taxvaluedollarcnt_county_mean_ratio',
    'taxvaluedollarcnt_county_std_ratio',
    'bathroomcnt_lat_lon_block_mean',
    'bathroomcnt_lat_lon_block_mean_ratio',
    'bathroomcnt_lat_lon_block_std',
    'bathroomcnt_lat_lon_block_std_ratio',
    'bedroomcnt_lat_lon_block_mean',
    'bedroomcnt_lat_lon_block_mean_ratio',
    'bedroomcnt_lat_lon_block_std',
    'bedroomcnt_lat_lon_block_std_ratio',
    'buildingqualitytypeid_lat_lon_block_mean',
    'buildingqualitytypeid_lat_lon_block_mean_ratio',
    'buildingqualitytypeid_lat_lon_block_std',
    'buildingqualitytypeid_lat_lon_block_std_ratio',
    'calculatedfinishedsquarefeet_lat_lon_block_mean',
    'calculatedfinishedsquarefeet_lat_lon_block_mean_ratio',
    'calculatedfinishedsquarefeet_lat_lon_block_std',
    'calculatedfinishedsquarefeet_lat_lon_block_std_ratio',
    'fullbathcnt_lat_lon_block_mean',
    'fullbathcnt_lat_lon_block_mean_ratio',
    'fullbathcnt_lat_lon_block_std',
    'fullbathcnt_lat_lon_block_std_ratio',
    'garagecarcnt_lat_lon_block_mean',
    'garagecarcnt_lat_lon_block_mean_ratio',
    'garagecarcnt_lat_lon_block_std',
    'garagecarcnt_lat_lon_block_std_ratio',
    'garagetotalsqft_lat_lon_block_mean',
    'garagetotalsqft_lat_lon_block_mean_ratio',
    'garagetotalsqft_lat_lon_block_std',
    'garagetotalsqft_lat_lon_block_std_ratio',
    'lotsizesquarefeet_lat_lon_block_mean',
    'lotsizesquarefeet_lat_lon_block_mean_ratio',
    'lotsizesquarefeet_lat_lon_block_std',
    'lotsizesquarefeet_lat_lon_block_std_ratio',
    'numberofstories_lat_lon_block_mean',
    'numberofstories_lat_lon_block_mean_ratio',
    'numberofstories_lat_lon_block_std',
    'numberofstories_lat_lon_block_std_ratio',
    'unitcnt_lat_lon_block_mean',
    'unitcnt_lat_lon_block_mean_ratio',
    'unitcnt_lat_lon_block_std',
    'unitcnt_lat_lon_block_std_ratio',
    'yearbuilt_lat_lon_block_mean',
    'yearbuilt_lat_lon_block_mean_ratio',
    'yearbuilt_lat_lon_block_std',
    'yearbuilt_lat_lon_block_std_ratio',
    'structuretaxvaluedollarcnt_lat_lon_block_mean',
    'structuretaxvaluedollarcnt_lat_lon_block_mean_ratio',
    'structuretaxvaluedollarcnt_lat_lon_block_std',
    'structuretaxvaluedollarcnt_lat_lon_block_std_ratio',
    'taxamount_lat_lon_block_mean',
    'taxamount_lat_lon_block_mean_ratio',
    'taxamount_lat_lon_block_std',
    'taxamount_lat_lon_block_std_ratio',
    'taxvaluedollarcnt_lat_lon_block_mean',
    'taxvaluedollarcnt_lat_lon_block_mean_ratio',
    'taxvaluedollarcnt_lat_lon_block_std',
    'taxvaluedollarcnt_lat_lon_block_std_ratio',
    'neighborhood_count',
    'bathroomcnt_neighborhood_mean',
    'bathroomcnt_neighborhood_std',
    'bathroomcnt_neighborhood_mean_ratio',
    'bedroomcnt_neighborhood_mean',
    'bedroomcnt_neighborhood_std',
    'bedroomcnt_neighborhood_mean_ratio',
    'buildingqualitytypeid_neighborhood_mean',
    'buildingqualitytypeid_neighborhood_std',
    'buildingqualitytypeid_neighborhood_mean_ratio',
    'calculatedfinishedsquarefeet_neighborhood_mean',
    'calculatedfinishedsquarefeet_neighborhood_std',
    'calculatedfinishedsquarefeet_neighborhood_mean_ratio',
    'fullbathcnt_neighborhood_mean',
    'fullbathcnt_neighborhood_std',
    'fullbathcnt_neighborhood_mean_ratio',
    'garagecarcnt_neighborhood_mean',
    'garagecarcnt_neighborhood_std',
    'garagecarcnt_neighborhood_mean_ratio',
    'unitcnt_neighborhood_mean',
    'unitcnt_neighborhood_std',
    'unitcnt_neighborhood_mean_ratio',
    'yearbuilt_neighborhood_mean',
    'yearbuilt_neighborhood_std',
    'yearbuilt_neighborhood_mean_ratio',
    'structuretaxvaluedollarcnt_neighborhood_mean',
    'structuretaxvaluedollarcnt_neighborhood_std',
    'structuretaxvaluedollarcnt_neighborhood_mean_ratio',
    'taxamount_neighborhood_mean',
    'taxamount_neighborhood_std',
    'taxamount_neighborhood_mean_ratio',
    'taxvaluedollarcnt_neighborhood_mean',
    'taxvaluedollarcnt_neighborhood_std',
    'taxvaluedollarcnt_neighborhood_mean_ratio',
    'zip_count',
    'bathroomcnt_zip_mean',
    'bathroomcnt_zip_std',
    'bathroomcnt_zip_mean_ratio',
    'bathroomcnt_zip_std_ratio',
    'bedroomcnt_zip_mean',
    'bedroomcnt_zip_std',
    'bedroomcnt_zip_mean_ratio',
    'bedroomcnt_zip_std_ratio',
    'buildingqualitytypeid_zip_mean',
    'buildingqualitytypeid_zip_std',
    'buildingqualitytypeid_zip_mean_ratio',
    'buildingqualitytypeid_zip_std_ratio',
    'calculatedfinishedsquarefeet_zip_mean',
    'calculatedfinishedsquarefeet_zip_std',
    'calculatedfinishedsquarefeet_zip_mean_ratio',
    'calculatedfinishedsquarefeet_zip_std_ratio',
    'fullbathcnt_zip_mean',
    'fullbathcnt_zip_std',
    'fullbathcnt_zip_mean_ratio',
    'fullbathcnt_zip_std_ratio',
    'garagecarcnt_zip_mean',
    'garagecarcnt_zip_std',
    'garagecarcnt_zip_mean_ratio',
    'garagecarcnt_zip_std_ratio',
    'unitcnt_zip_mean',
    'unitcnt_zip_std',
    'unitcnt_zip_mean_ratio',
    'unitcnt_zip_std_ratio',
    'yearbuilt_zip_mean',
    'yearbuilt_zip_std',
    'yearbuilt_zip_mean_ratio',
    'yearbuilt_zip_std_ratio',
    'structuretaxvaluedollarcnt_zip_mean',
    'structuretaxvaluedollarcnt_zip_std',
    'structuretaxvaluedollarcnt_zip_mean_ratio',
    'structuretaxvaluedollarcnt_zip_std_ratio',
    'taxamount_zip_mean',
    'taxamount_zip_std',
    'taxamount_zip_mean_ratio',
    'taxamount_zip_std_ratio',
    'taxvaluedollarcnt_zip_mean',
    'taxvaluedollarcnt_zip_std',
    'taxvaluedollarcnt_zip_mean_ratio',
    'taxvaluedollarcnt_zip_std_ratio',
    'multiply_lat_lon',
    'poly_2_structure_tax_value',
    'poly_3_structure_tax_value',
    'ratio_basement',
    'ratio_bedroom_bathroom',
    'ratio_fireplace',
    'ratio_floor_shape',
    'ratio_living_area',
    'ratio_living_area_2',
    'ratio_pool_shed',
    'ratio_pool_yard',
    'ratio_structure_tax_value_to_land_tax_value',
    'ratio_tax',
    'ratio_tax_value_to_land_tax_value',
    'ratio_tax_value_to_structure_value',
    'round_lat',
    'round_lon',
    'sum_lat_lon',
    'total_rooms',
    'target_neighborhood_feature',
    'target_zip_feature',
    'target_city_feature',
    'target_county_feature',
    'poolcnt',
    'logerror_regionidneighborhood_max',
    'logerror_regionidneighborhood_min',
    'logerror_regionidneighborhood_std',
    'logerror_regionidneighborhood_mean',
    'logerror_regionidneighborhood_std_over_mean',
    'logerror_regionidneighborhood_range',
    'logerror_regionidzip_max',
    'logerror_regionidzip_min',
    'logerror_regionidzip_std',
    'logerror_regionidzip_mean',
    'logerror_regionidzip_std_over_mean',
    'logerror_regionidzip_range',
    'logerror_regionidcity_max',
    'logerror_regionidcity_min',
    'logerror_regionidcity_std',
    'logerror_regionidcity_mean',
    'logerror_regionidcity_std_over_mean',
    'logerror_regionidcity_range',
    'logerror_regionidcounty_max',
    'logerror_regionidcounty_min',
    'logerror_regionidcounty_std',
    'logerror_regionidcounty_mean',
    'logerror_regionidcounty_std_over_mean',
    'logerror_regionidcounty_range',
    'logerror_fips_census_1_max',
    'logerror_fips_census_1_min',
    'logerror_fips_census_1_std',
    'logerror_fips_census_1_mean',
    'logerror_fips_census_1_std_over_mean',
    'logerror_fips_census_1_range',
    'logerror_fips_census_block_max',
    'logerror_fips_census_block_min',
    'logerror_fips_census_block_std' ,
    'logerror_fips_census_block_mean' ,
    'logerror_fips_census_block_std_over_mean',
    'logerror_fips_census_block_range',
]

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

def prepare_features(year, feature_list, clean=False):
    # use minimized version of properties data when memory is a concern
    # prop = utils.load_properties_data_minimize()
    # feature engineering
    print('Feature engineering, year %d' %year)
    if clean:
        print('Using cleaned prop')
        prop = feature_combine.feature_combine_cleaned(year, feature_list)
    else:
        print('Using prop with nan')
        prop = feature_combine.feature_combine_with_nan(year, feature_list)
    print('prop shape:')
    print(prop.shape)
    # for col in prop.columns:
    #   print('\'' + col + '\',')
    # print('Nan cells: ' + str(prop.isnull().sum().sum()))

    return prop

def prepare_training_data(year, prop):
    # load training data
    print('Load transaction data...')
    transactions = utils.load_transaction_data(year)

    # merge transaction and prop data
    train_df = transactions.merge(prop, how='left', on='parcelid')

    return train_df

def predict_cap(predict, thres=1.5):
    print('Clip out output abs value greater than %s' % thres)
    predict[predict>thres] = thres
    predict[predict<-thres] = -thres
    return predict

def train(X_train, y_train, X_validate, y_validate, X_test,
    Model, model_params = None,
    outliers_up_pct = 100, outliers_lw_pct = 0,
    submit=False,
    scaling=False, scaler=RobustScaler(quantile_range=(0, 99)), scaling_columns=SCALING_COLUMNS,
    return_models=False):

    # Get and drop id columns for validate and test
    X_validate_id = X_validate['parcelid']
    X_validate.drop('parcelid', axis=1, inplace=True)
    if submit:
        X_test_id = X_test['parcelid']
        X_test.drop('parcelid', axis=1, inplace=True)

    print('Train dimensions: ')
    print(X_train.shape, y_train.shape)

    print('Validation dimensions: ')
    print(X_validate.shape, y_validate.shape)

    if submit:
        print('Test dimensions: ')
        print(X_test.shape)

    # Scaling train and test features if needed
    if scaling:
        for col in X_train.columns:
            if col in scaling_columns:
                # fit and transform training data
                X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                X_validate[col] = scaler.transform(X_validate[col].values.reshape(-1, 1))
                if submit:
                    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

    ulimit = np.percentile(y_train.values, outliers_up_pct)
    llimit = np.percentile(y_train.values, outliers_lw_pct)
    mask = (y_train >= llimit) & (y_train <= ulimit)
    print(llimit, ulimit)
    X_train_no_outlier = X_train[mask]
    y_train_no_outlier = y_train[mask]
    print(X_train_no_outlier.shape, y_train_no_outlier.shape)

    print('training...')
    model = Model(model_params = model_params)
    model.fit(X_train_no_outlier, y_train_no_outlier)
    pred_train = model.predict(X_train_no_outlier)
    print("fold train mean error: ", Evaluator.mean_error(pred_train, y_train_no_outlier))

    print('validating...')
    pred_validate = model.predict(X_validate)
    mae_validate = Evaluator.mean_error(pred_validate, y_validate)
    print("fold validation mean error: ", mae_validate)
    # match validation predictions with its parcelid and wrap them into a series
    pred_validate = pd.Series(pred_validate, index=X_validate_id)

    
    if submit:
        print('Re-train model on the entire dataset')
        X_train_all = pd.concat([X_train, X_validate])
        y_train_all = pd.concat([y_train, y_validate])
        print('All training data shape')
        print(X_train_all.shape, y_train_all.shape)

        ulimit = np.percentile(y_train_all.values, outliers_up_pct)
        llimit = np.percentile(y_train_all.values, outliers_lw_pct)
        mask = (y_train_all >= llimit) & (y_train_all <= ulimit)
        print(llimit, ulimit)
        X_train_all_no_outlier = X_train_all[mask]
        y_train_all_no_outlier = y_train_all[mask]
        print(X_train_all_no_outlier.shape, y_train_all_no_outlier.shape)

        model.fit(X_train_all_no_outlier, y_train_all_no_outlier)

        # TODO: add transactiondate features
        print('predicting on testing data...')
        test_preds = []
        # Split testing dataframe
        for X_test_split in np.array_split(X_test, 30):
            test_preds.append(model.predict(X_test_split))
        pred_test = np.concatenate(test_preds)

        pred_test = pd.Series(pred_test, index=X_test_id)
        print('prediction length: %d' %len(avg_pred))
        print('nan in predictions: %d' %pred_test.isnull().sum())
        print(pred_test.describe())

        if scaling:
            pred_test = predict_cap(pred_test, y_train_all.abs().max())

    if not submit:
        pred_test = None
    # whether return models for investigation
    if not return_models:
        model = None
    return mae_validate, pred_validate, pred_test, model

def train_wrapper(df2016, df_all, Model, params,
        mode, submit=False, prop2016=None, prop2017=None, config_name='no_name'):

    if mode == 'tune':
        submit = False
        params['return_models'] = False


    if mode == 'return_models':
        params['return_models'] = True

    # 2016 training
    df_train2016, df_validate2016 = utils.split_by_date(df2016, '2016-10-01')
    df_train2016.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    df_validate2016.drop('transactiondate', axis=1, inplace=True)
    X_train2016, y_train2016 = utils.get_features_target(df_train2016)
    X_validate2016, y_validate2016 = utils.get_features_target(df_validate2016)

    mae_validate2016, pred_validate2016, pred_test2016, model2016 = train(
        X_train2016, y_train2016, X_validate2016, y_validate2016, prop2016,
        Model, **params,
        submit=submit)

    # 2016 - 2017 training
    df_train_all, df_validate_all = utils.split_by_date(df_all, '2017-07-01')
    df_train_all.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    df_validate_all.drop('transactiondate', axis=1, inplace=True)
    X_train_all, y_train_all = utils.get_features_target(df_train_all)
    X_validate_all, y_validate_all = utils.get_features_target(df_validate_all)

    mae_validate_all, pred_validate_all, pred_test_all, model_all = train(
        X_train_all, y_train_all, X_validate_all, y_validate_all, prop2017,
        Model, **params,
        submit=submit)

    print('2016 validate mae: %f' %mae_validate2016)
    print('all validate mae: %f' %mae_validate_all)

    average_mae = (mae_validate2016 + mae_validate_all) / 2
    print('average mae: %f' %average_mae)

    if mode == 'tune':
        return average_mae

    if mode == 'stacking':
        validate_folder = 'data/ensemble/csv/validate/'
        if not os.path.exists(validate_folder):
            os.makedirs(validate_folder)
        pred_validate2016.to_csv('data/ensemble/csv/validate/%s2016.csv' %config_name)
        pred_validate_all.to_csv('data/ensemble/csv/validate/%s_all.csv' %config_name)

    # TODO: add transactiondate features to the generatio of submission
    if submit:
        test_folder = 'data/ensemble/csv/test/'
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        pred_test2016.to_csv('data/ensemble/csv/test/%s2016.csv' %config_name)
        pred_test_all.to_csv('data/ensemble/csv/test/%s_all.csv' %config_name)
        # TODO: generate another copy for submission use

    if mode == 'return_models':
        return [model2016, model_all]



def get_dfs(config_dict, include_properties=False):

    # Feature list
    feature_list = config_dict['feature_list']
    # clean_na
    clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False

    prop2016 = prepare_features(2016, feature_list, clean_na)
    df2016 = prepare_training_data(2016, prop2016)

    prop2017 = prepare_features(2017, feature_list, clean_na)
    df2017 = prepare_training_data(2017, prop2017)

    df_all = pd.concat([df2016, df2017])

    if not include_properties:
        prop2016 = None
        prop2017 = None
        gc.collect()

    return df2016, df_all, prop2016, prop2017


if __name__ == '__main__':
    t1 = time.time()
    # Get configuration
    # parser to parse cmd line option
    parser = OptionParser()
    # add options to parser, currently only config file
    parser.add_option('-c', '--config', action='store', type='string', dest='config_file')
    parser.add_option('-s', '--submit', action='store_true', dest='submit', default=False)
    # parse cmd line arguments
    (options, args) = parser.parse_args()

    config_file = options.config_file
    # if generate a submission
    submit = options.submit

    # Configuration:
    config_dict = getattr(config, config_file)
    # print configuration for confirmation
    for key, value in config_dict.items():
        if key == 'feature_list':
            for k, v in value.items():
                print('%s: %s' %(k, len(v)))
        elif key == 'stacking_params' or key == 'tuning_params':
            continue
        else:
            print('%s: %s' %(key, value))
    print('submit: %s' %submit)

    df2016, df_all, prop2016, prop2017 = get_dfs(config_dict, submit)
    # # model
    Model = config_dict['Model']
    params = config_dict['training_params']
    train_wrapper(df2016, df_all, Model, params,
        'stacking', submit, prop2016, prop2017, config_dict['name'])

    t2 = time.time()
    print((t2 - t1) / 60)
