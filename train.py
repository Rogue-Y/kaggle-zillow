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

def prepare_features(feature_list = [], clean=False):
    # use minimized version of properties data when memory is a concern
    # prop = utils.load_properties_data_minimize()
    # feature engineering
    print('Feature engineering')
    if clean:
        print('Using cleaned prop')
        prop = feature_combine.feature_combine_cleaned(feature_list)
    else:
        print('Using prop with nan')
        prop = feature_combine.feature_combine_with_nan(feature_list)
    print(prop.shape)
    # for col in prop.columns:
    #   print('\'' + col + '\',')
    # print('Nan cells: ' + str(prop.isnull().sum().sum()))

    return prop

def prepare_training_data(prop):
    # Process:
    # load training data
    print('Load training data...')
    transactions = utils.load_transaction_data()

    # merge transaction and prop data
    train_df = transactions.merge(prop, how='left', on='parcelid')
    # df.to_csv('test_df.csv')
    # del train_df; gc.collect()

    return train_df, transactions

def train(train_df, Model, model_params = None, FOLDS = 5, record=False,
    outliers_up_pct = 99, outliers_lw_pct = 1,
    submit=False, prop = None, transactions = None, # if submit is true, than must provide transactions and prop
    resale_offset = 0.012, pca_components=-1, scaling=False, scaler=RobustScaler(quantile_range=(0, 99)), scaling_columns=SCALING_COLUMNS):
    # Optional dimension reduction.
    if pca_components > 0:
        print('PCA...')
        pca = PCA(n_components=pca_components)
        feature_df, non_feature_df = utils.get_dimension_reduction_df(train_df)
        # Note that the features pca produces is some combination of the original features, not retain/discard some columns
        feature_df = pca.fit_transform(feature_df)
        train_df = pd.concat([pd.DataFrame(feature_df), non_feature_df], axis=1)

    print('Train df dimensions: ' + str(train_df.shape))
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

        X_validate = X_train_q4.loc[validate_index]
        y_validate = y_train_q4.loc[validate_index]

        # Fit scalers using the training data and transform both training, validating and testing data
        if scaling:
            print('Scaling...')
            for col in scaling_columns:
                if col in X_train.columns:
                    # fit and transform training data
                    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                    # transform both validating and testing data
                    X_validate[col] = scaler.transform(X_validate[col].values.reshape(-1, 1))
                    if submit:
                        df_test[col] = scaler.transform(df_test[col].values.reshape(-1, 1))

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
    # scaling
    scaling = config_dict['scaling'] if 'scaling' in config_dict else False
    # PCA component
    pca_components = config_dict['pca_components'] if 'pca_components' in config_dict else -1
    # PCA cannot deal with infinite numbers
    if pca_components > 0:
        clean_na = True


    prop = prepare_features(feature_list, clean_na)

    train_df, transactions = prepare_training_data(prop)
    if submit:
        cv_error = train(train_df,
            prop=prop, transactions=transactions, Model=Model, model_params=model_params, FOLDS = FOLDS,
            record=record, submit=True, outliers_up_pct=outliers_up_pct,
            outliers_lw_pct=outliers_lw_pct, resale_offset=resale_offset, pca_components=pca_components, scaling=scaling)
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
            outliers_lw_pct=outliers_lw_pct, resale_offset=resale_offset, pca_components=pca_components, scaling=scaling)

    t2 = time.time()
    print((t2 - t1) / 60)
