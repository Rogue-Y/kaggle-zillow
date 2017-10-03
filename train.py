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

def prepare_features(feature_list, clean=False):
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

def predict_cap(predict, thres=1.5):
    print('Clip out output abs value greater than %s' % thres)
    predict[predict>thres] = thres
    predict[predict<-thres] = -thres
    return predict

def train(train_df, Model, model_params = None, FOLDS = 5, record=False,
    outliers_up_pct = 100, outliers_lw_pct = 0,
    submit=False, prop = None, transactions = None, config_dict={}, # if submit is true, than must provide transactions and prop
    resale_offset = 0.012, pca_components=-1,
    scaling=False, scaler=RobustScaler(quantile_range=(0, 99)), scaling_columns=SCALING_COLUMNS,
    return_models=False):
    # Optional dimension reduction.
    if pca_components > 0:
        print('PCA...')
        pca = PCA(n_components=pca_components)
        feature_df, non_feature_df = utils.get_dimension_reduction_df(train_df)
        # Note that the features pca produces is some combination of the original features, not retain/discard some columns
        feature_df = pca.fit_transform(feature_df)
        train_df = pd.concat([pd.DataFrame(feature_df), non_feature_df], axis=1)

    # If need to generate submission, prepare prop df for testing
    if submit:
        # When we clean data, we removed the rows where lat and lon are null,
        # so we have to preserve the parcelid of the prop here to join with that
        # of the sample submission
        # training data(transactions) don't have this problem as all of them
        # have valid lat and lon
        df_test_parcelid = prop['parcelid']
        df_test = data_clean.drop_id_column(prop)
    else:
        del prop; gc.collect()

    # Scaling train and test features if needed
    if scaling:
        for col in train_df.columns:
            if col in scaling_columns:
                # fit and transform training data
                train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1))
                if submit:
                    df_test[col] = scaler.transform(df_test[col].values.reshape(-1, 1))

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

    # split train_q4 into k folds, each time combine k-1 folds with train_q1_q3
    # to train model and validate on the left out fold
    mean_errors = []
    model_preds = []
    if return_models:
        models = []
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    # X_train_q4.to_csv('test_train_q4.csv')
    for i, (train_index, validate_index) in enumerate(kf.split(X_train_q4)):
        X_train = pd.concat([X_train_q1_q3, X_train_q4.iloc[train_index]], ignore_index=True)
        y_train = pd.concat([y_train_q1_q3, y_train_q4.iloc[train_index]], ignore_index=True)
        # TODO(hzn): add training preprocessing, like remove outliers, resampling

        X_validate = X_train_q4.loc[validate_index]
        y_validate = y_train_q4.loc[validate_index]

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
            print(pd.DataFrame(np.concatenate(test_preds)).describe())
            model_preds.append(np.concatenate(test_preds))
        if return_models:
            models.append(model)
        print("--------------------------------------------------------")

    avg_cv_errors = np.mean(mean_errors)
    if record:
        train_recorder.write("\nAverage cross validation mean error: %f\n" %avg_cv_errors)
        train_recorder.close()
    print("average cross validation mean error", avg_cv_errors)

    if submit:
        print("loading submission data...")
        predict_df, sample = utils.load_test_data()
        print(predict_df.shape)
        print(sample.shape)

        # make prediction
        print("make prediction...")
        # model_preds = list(map(lambda model: model.predict(df_test), models))
        avg_pred = pd.Series(np.mean(model_preds, axis=0), name='predict', index=df_test_parcelid)
        print('prediction length: %d' %len(avg_pred))

        # add resale
        sales = transactions[['parcelid', 'logerror']].groupby('parcelid').mean()
        predict_df = predict_df.join(sales, on='parcelid', how='left')
        predict_df = predict_df.join(avg_pred, on='parcelid', how='left')
        # predict = predict_df['predict'].where(predict_df['logerror'].isnull(), predict_df['logerror'])
        predict = predict_df['predict'].where(
            predict_df['logerror'].isnull(), predict_df['predict'] + resale_offset)
        # Sanity check
        print('nan in predictions: %d' %predict.isnull().sum())
        # For those we do not predict (parcels whose lat and lon are nan), fill
        # the median logerror
        predict.fillna(0.011, inplace=True)
        # Scaling could cause the model to have some wiredly large predictions
        # cap them with largest abs in the trainning logerror
        if scaling:
            predict = predict_cap(predict, max(y_train_q4.abs().max(), y_train_q1_q3.abs().max()))

        # Save prediction(a Series object) to a pickle for later use
        # Save one in history
        sub_history_folder = 'data/predictions/history'
        if not os.path.exists(sub_history_folder):
            os.makedirs(sub_history_folder)
        predict.to_pickle('%s/%s_%s_pickle' %(sub_history_folder, time, Model.__name__))
        # # Update the most recent pickle for this model
        # predict.to_pickle('data/predictions/%s_latest_pickle' %Model.__name__)

        # generate submission
        print("generating submission...")
        for c in sample.columns[sample.columns != 'ParcelId']:
            # sample[c] = avg_pred
            sample[c] = predict.as_matrix()
        # time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        submission_folder = 'data/submissions'
        if not os.path.exists(submission_folder):
            os.makedirs(submission_folder)
        sample.to_csv(
            '%s/Submission_%s.csv' %(submission_folder, time), index=False, float_format='%.4f')
        print("Prediction made.")

        exp_record_folder = 'data/experiments'
        if not os.path.exists(exp_record_folder):
            os.makedirs(exp_record_folder)
        with open('%s/%s.txt' %(exp_record_folder, 'experiments'), 'a') as record:
             # Time is the same across submission csv, pickle and record for easy search
            record.write('\n\nTime: %s\n' %time)
            record.write('Config: %s\n' %config_dict)
            record.write('cv_error:%s\n' %avg_cv_errors)
            record.write('leaderboard:________________PLEASE FILL____________________\n')

    # if return models for investigation
    if return_models:
        return avg_cv_errors, models

    return avg_cv_errors

def train_stacking(train_df, Model, model_params = None, FOLDS = 5,
    outliers_up_pct = 100, outliers_lw_pct = 0,
    submit=False, config_name='', prop = None,  # if submit is true, than must provide prop and config_name
    pca_components=-1, scaling=False, scaler=RobustScaler(quantile_range=(0, 99)),
    scaling_columns=SCALING_COLUMNS):
    # Optional dimension reduction.
    if pca_components > 0:
        print('PCA...')
        pca = PCA(n_components=pca_components)
        feature_df, non_feature_df = utils.get_dimension_reduction_df(train_df)
        # Note that the features pca produces is some combination of the original features, not retain/discard some columns
        feature_df = pca.fit_transform(feature_df)
        train_df = pd.concat([pd.DataFrame(feature_df), non_feature_df], axis=1)

    # If need to generate submission, prepare prop df for testing
    if submit:
        # When we clean data, we removed the rows where lat and lon are null,
        # so we have to preserve the parcelid of the prop here to join with that
        # of the sample submission
        # training data(transactions) don't have this problem as all of them
        # have valid lat and lon
        df_test_parcelid = prop['parcelid']
        df_test = data_clean.drop_id_column(prop)
    else:
        del prop; gc.collect()

    # Scale train and test features if needed.
    if scaling:
        for col in train_df.columns:
            if col in scaling_columns:
                # fit and transform training data
                train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1))
                if submit:
                    df_test[col] = scaler.transform(df_test[col].values.reshape(-1, 1))


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

    # split train_q4 into k folds, each time combine k-1 folds with train_q1_q3
    # to train model and validate on the left out fold
    model_preds = [] # predictions on test set
    validate_fold_preds = [] # predictions on validation set
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    # X_train_q4.to_csv('test_train_q4.csv')
    for i, (train_index, validate_index) in enumerate(kf.split(X_train_q4)):
        X_train = pd.concat([X_train_q1_q3, X_train_q4.iloc[train_index]], ignore_index=True)
        y_train = pd.concat([y_train_q1_q3, y_train_q4.iloc[train_index]], ignore_index=True)
        # TODO(hzn): add training preprocessing, like remove outliers, resampling

        X_validate = X_train_q4.loc[validate_index]
        y_validate = y_train_q4.loc[validate_index]

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
        # train_pred = model.predict(X_train)
        # print("fold train mean error: ", Evaluator.mean_error(train_pred, y_train))

        print('validating...')
        y_pred = model.predict(X_validate)
        validate_fold_preds.append(pd.Series(y_pred, index=validate_index))
        # Predict on testing data if needed to generate submission
        if submit:
            print('predicting on testing data...')
            test_preds = []
            # Split testing dataframe
            for df_test_split in np.array_split(df_test, 30):
                test_preds.append(model.predict(df_test_split))
            model_preds.append(np.concatenate(test_preds))
        print("--------------------------------------------------------")

    pred_col_name = 'predict_%s' %config_name
    validation_pred = pd.concat(validate_fold_preds).sort_index().rename(pred_col_name)
    if submit:
        # need to merge to the full properties set and fill na for the cleaned
        # data set
        test_pred = pd.Series(np.mean(model_preds, axis=0), name=pred_col_name, index=df_test_parcelid)
        print('Prediction describe for %s' %config_name)
        print(test_pred.describe())
        prop_full = utils.load_properties_data_raw()
        test_pred = prop_full.join(test_pred, on='parcelid', how='left')[pred_col_name]
        # Sanity check
        print('nan in predictions: %d' %test_pred.isnull().sum())
        test_pred.fillna(0.011, inplace=True)
    else:
        test_pred = None
    return validation_pred, y_train_q4, test_pred # this is None when not submitting

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

    # Mandatory configurations:
    # Feature list
    feature_list = config_dict['feature_list']
    # # model
    Model = config_dict['Model']
    # clean_na
    clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False

    prop = prepare_features(feature_list, clean_na)

    train_df, transactions = prepare_training_data(prop)
    if submit:
        cv_error = train(train_df, Model=Model,
            submit=True, prop=prop, transactions=transactions,
            config_dict=config_dict, # config dict for record submission purpose
            **config_dict['training_params'])
    else:
        del transactions; del prop; gc.collect()
        _ , models = train(train_df, Model=Model, submit=False, return_models=True, **config_dict['training_params'])
        utils.dump_aux(models, 'BestRidgeModels')

    t2 = time.time()
    print((t2 - t1) / 60)
