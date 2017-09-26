from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import inspect
import json
import os
import pickle

from features import feature_clean

# utility functions
def plot_score(y, y_hat):
    return (np.mean(abs(y-y_hat)))

def load_train_data(data_folder='data/', force_read=False):
    """ Load transaction data and properties data.
        Returns:
            (train_df, properties_df)
    """
    train = load_transaction_data(data_folder, force_read)
    prop = load_properties_data(data_folder, force_read)

    return (train, prop)

def load_transaction_data(data_folder='data/', force_read=False):
    """ Load transaction data.
        Returns:
            train_df
    """
    train_data_pickle = data_folder + 'train_2016_v2_pickle'

    if not force_read and os.path.exists(train_data_pickle):
        train = pd.read_pickle(train_data_pickle)
    else:
        train = pd.read_csv(
            data_folder + 'train_2016_v2.csv', parse_dates=['transactiondate'])
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        train.to_pickle(train_data_pickle)
    return train

def load_properties_data(data_folder='data/', force_read=False):
    """ Load properties data.
        Returns:
            properties_df
    """
    prop_data_pickle = data_folder + 'properties_2016_pickle'

    if not force_read and os.path.exists(prop_data_pickle):
        prop = pd.read_pickle(prop_data_pickle)
    else:
        prop = pd.read_csv(data_folder + 'properties_2016.csv')
        # Fill missing geo data a little bit
        prop = preprocess_geo(prop)
        # Convert float64 to float32 to save memory
        for col in prop.columns:
            if prop[col].dtype == 'float64':
                prop[col] = prop[col].astype('float32')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        prop.to_pickle(prop_data_pickle)
    return prop


def load_properties_data_preprocessed(data_folder='data/', force_read=False):
    """ Load properties data and do some simple preprocess, mainly on boolean
        and categorical data.
        Returns:
            properties_df_preprocessed
    """
    prop_preprocessed_pickle_path = data_folder + 'properties_2016_pickle_preprocessed'
    if not force_read and os.path.exists(prop_preprocessed_pickle_path):
        prop_preprocessed = pd.read_pickle(prop_preprocessed_pickle_path)
    else:
        prop_preprocessed = load_properties_data()

        # Preprocessing some columns so that all columns are numbers/booleans and has concrete meanings
        # boolean columns
        # prop_preprocessed['fireplaceflag'] = feature_clean.fireplacecnt(prop_preprocessed)
        # prop_preprocessed['hashottuborspa'] = feature_clean.hashottuborspa(prop_preprocessed)
        # prop_preprocessed['pooltypeid10'] = feature_clean.pooltypeid10(prop_preprocessed)
        # prop_preprocessed['pooltypeid2'] = feature_clean.pooltypeid2(prop_preprocessed)
        # prop_preprocessed['pooltypeid7'] = feature_clean.pooltypeid7(prop_preprocessed)
        # prop_preprocessed['storytypeid'] = feature_clean.storytypeid(prop_preprocessed)
        # prop_preprocessed['decktypeid'] = feature_clean.decktypeid(prop_preprocessed)
        # prop_preprocessed['poolcnt'] = feature_clean.poolcnt(prop_preprocessed)
        prop_preprocessed['taxdelinquencyflag'] = feature_clean.taxdelinquencyflag(prop_preprocessed)

        # change taxdelinquencyyear to YYYY format
        tmp = prop_preprocessed['taxdelinquencyyear'] + 1900
        prop_preprocessed['taxdelinquencyyear'] = tmp.where(
            prop_preprocessed['taxdelinquencyyear'] > 17,
            tmp + 100)

        # encode some categorical features that are not represented with numbers
        labelEncoder = LabelEncoder()
        prop_preprocessed['propertycountylandusecode'] = (
            labelEncoder.fit_transform(prop_preprocessed['propertycountylandusecode'].astype(str)))
        prop_preprocessed['propertyzoningdesc'] = (
            labelEncoder.fit_transform(prop_preprocessed['propertyzoningdesc'].astype(str)))

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        prop_preprocessed.to_pickle(prop_preprocessed_pickle_path)

    return prop_preprocessed


def load_properties_data_cleaned(data_folder='data/', force_read=False):
    """ Load properties data and clean data.
        Returns:
            properties_df_cleaned
    """
    prop_cleaned_pickle_path = data_folder + 'properties_2016_pickle_cleaned'
    if not force_read and os.path.exists(prop_cleaned_pickle_path):
        prop_cleaned = pd.read_pickle(prop_cleaned_pickle_path)
    else:
        # (name, function)
        functions = [o for o in inspect.getmembers(feature_clean) if inspect.isfunction(o[1])]
        # convert to a dictionary
        functions_dict = dict(functions)
        prop = load_properties_data()
        # filter out parcels that have unknown lat or lon
        prop = prop[prop['latitude'].notnull() & prop['longitude'].notnull()]
        # create a new df as some feature filling may depend on other features
        prop_cleaned = pd.DataFrame()
        for col in prop.columns:
            prop_cleaned[col] = functions_dict[col](prop)

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        prop_cleaned.to_pickle(prop_cleaned_pickle_path)

    return prop_cleaned

# Load the reduced size properties data
def load_properties_data_minimize(data_folder='data/', force_read=False):
    """ Load properties data.
        Returns:
            properties_df
    """
    prop_data_pickle = data_folder + 'properties_2016_pickle_mini'

    if not force_read and os.path.exists(prop_data_pickle):
        prop = pd.read_pickle(prop_data_pickle)
    else:
        prop = pd.read_csv(data_folder + 'properties_2016.csv')
        # Fill missing geo data a little bit
        prop = preprocess_geo(prop)
        # Convert float64 to float32 to save memory
        prop, na_list = reduce_mem_usage(prop)
        prop.to_pickle(prop_data_pickle)
    return prop

def preprocess_geo(prop):
    """ Preprocess data, fill some missing regionidcity, zip and neighborhood
        from existing data.
        Returns:
            properties_df
    """
    geo = prop[["latitude", "longitude", "fips", "regionidcounty", "regionidcity",
        "regionidzip", "regionidneighborhood"]].copy()
    geo['missing'] = geo.isnull().sum(axis=1)
    geo_complete = geo[geo['missing'] == 0].copy()
    # TODO(hzn): the way this join key works depend on how the latitude and longitude
    # are convert to strings, in the float32 case, the key looks like:
    # 3.41444e+07-1.18654e+08
    # This means, every points in the square of (34.144400, -118.654000) and
    # (34.144499, -118.654999) will be redeemed as the same location.
    # The diagonal of such grouping is around 100m, which is reasonable to me,
    # but it somehow hurt the validation performance of the two models we have
    # a little bit. Therefore, more observations/experiements are needed.
    # On the other hand, in the float64 case, the join_key is formed with exact
    # lat/lon number: 34144442.0-118654084.0
    # so only exact same location are joined (probably two parcels in the same
    # building).
    # This one seems benefit lightgbm a little bit while hurt xgboost. Those changes
    # Could all due to the parameters is not optimal.
    # Besides filling missing geo, this is also an interesting feature
    # to investigate.
    geo['join_key'] = geo['latitude'].astype(str) + geo['longitude'].astype(str)
    geo_complete['join_key'] = geo_complete['latitude'].astype(str) + geo_complete['longitude'].astype(str)
    geo_complete = geo_complete.drop_duplicates('join_key').set_index('join_key')
    geo_join = geo.join(geo_complete, 'join_key', 'left', rsuffix = '_r')
    cols = ["regionidcity", "regionidzip", "regionidneighborhood"]
    for col in cols:
        prop[col] = geo_join[col].where(geo_join[col].notnull(), geo_join[col+'_r'])
    return prop

def load_test_data(data_folder='data/', force_read=False):
    """ Load data and join trasaction data with properties data.
        Returns:
            (joined_test_df, sample_submission_df)
    """
    test_data_pickle = data_folder + 'sample_submission_pickle'
    if not force_read and os.path.exists(test_data_pickle):
        sample = pd.read_pickle(test_data_pickle)
    else:
        sample = pd.read_csv(data_folder + 'sample_submission.csv')
        sample.to_pickle(test_data_pickle)
    # sample submission use "ParcelId" instead of "parcelid"
    test = sample.rename(index=str, columns={'ParcelId': 'parcelid'})
    # drop the month columns in sample submission
    test.drop(['201610', '201611', '201612', '201710', '201711', '201712'],
        axis=1, inplace=True)
    return (test, sample)

def load_sample_submission(data_folder='data/', force_read=False):
    """ Load data and join trasaction data with properties data.
        Returns:
            sample_submission_df
    """
    test_data_pickle = data_folder + 'sample_submission_pickle'
    if not force_read and os.path.exists(test_data_pickle):
        sample = pd.read_pickle(test_data_pickle)
    else:
        sample = pd.read_csv(data_folder + 'sample_submission.csv')
        sample.to_pickle(test_data_pickle)
    return sample

def load_config(config_file='config/steps.json'):
    """ Read config file
        Returns: config JSON object
    """
    config_file = open(config_file, "r")
    config = None
    try:
        config = json.load(config_file)
    finally:
        config_file.close()
    return config


def train_valid_split(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def get_features_target(df):
    """ Get features dataframe, and target column
        Call clean data and drop column in data_clean.py function before use
        this method.
        Returns:
            (X, y)
    """
    # logerror is the target column
    target = df['logerror']
    df.drop(['logerror'], axis=1, inplace=True)
    return (df, target)

def get_dimension_reduction_df(df):
    """ Get all feature columns for dimension reduction use.
        Returns:
            (feature_columns, other columns)
    """
    # non-feature columns, in case need to put them back after dimension reduction
    non_feature_columns = ['parcelid', 'transactiondate', 'logerror']
    non_feature_df = df[non_feature_columns]
    df.drop(non_feature_columns, axis=1, inplace=True)
    return (df, non_feature_df)

def split_by_date(df, split_date = '2016-10-01'):
    """ Split the transaction data into two part, those before split_date as
        training set, those after as test set.
        Returns:
            (train_df, test_df)
    """
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    # 82249 rows
    # loc is used here to get a real slice rather then a view, so there will not
    # be problem when trying to write to them.
    train_df = (df.loc[df['transactiondate'] < split_date, :]).reset_index(drop=True)
    # 8562 rows
    test_df = (df.loc[df['transactiondate'] >= split_date, :]).reset_index(drop=True)
    return (train_df, test_df)

def predict(predictor, train_cols):
    sample = pd.read_csv('sample_submission.csv')
    prop = pd.read_csv('properties_2016.csv')
    sample['parcelid'] = sample['ParcelId']
    df_test = sample.merge(prop, on='parcelid', how='left')
    x_test = df_test[train_cols]
    for c in x_test.dtypes[x_test.dtypes == object].index.values:
        x_test[c] = (x_test[c] == True)
    predictor.reset_parameter({"num_threads":1})
    p_test = predictor.predict(x_test)
    sub = pd.read_csv('sample_submission.csv')
    for c in sub.columns[sub.columns != 'ParcelId']:
        sub[c] = p_test

    sub.to_csv('lgb_starter.csv', index=False, float_format='%.4f')

def predict(predictor, X_test, sample, suffix=''):
    """ Predict on test set and write to a csv
        Params:
            predictor - the predictor, using lightgbm now
            X_test - the test features dataframe
            sample - sample_submission dataframe
            suffix - suffix of output file
    """
    predictor.reset_parameter({"num_threads":1})
    p_test = predictor.predict(X_test)
    for c in sample.columns[sample.columns != 'ParcelId']:
        sample[c] = p_test

    sample.to_csv('data/lgb_starter'+suffix+'.csv', index=False, float_format='%.4f')

# def get_train_test_sets():
#     """ Get the training and testing set: now split by 2016-10-01
#         transactions before this date serve as training data; those after as
#         test data.
#         Returns:
#             (X_train, y_train, X_test, y_test)
#     """
#     print('Loading data ...')
#     train, prop = load_train_data()
#
#     print('Cleaning data and feature engineering...')
#     prop_df = feature_eng.add_missing_value_count(prop)
#     prop_df = data_clean.clean_categorical_data(prop_df)
#
#     # Subset with transaction info
#     df = train.merge(prop_df, how='left', on='parcelid')
#     df = feature_eng.convert_year_build_to_age(df)
#     df = data_clean.drop_low_ratio_columns(df)
#     df = data_clean.clean_boolean_data(df)
#     df = data_clean.drop_id_column(df)
#     df = data_clean.fillna(df)
#
#     print("Spliting data into training and testing...")
#     train_df, test_df = split_by_date(df)
#     # 82249 rows
#     X_train, y_train = get_features_target(train_df)
#     # 8562 rows
#     X_test, y_test = get_features_target(test_df)
#
#     print("Done")
#     return (X_train, y_train, X_test, y_test)

# Reduce properties dataset size
# From: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()


            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True


            # Make Integer/unsigned Integer datatypes
            if IsInt:
                # Numpy integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(props[col]).all():
                    NAlist.append(col)
                    props[col].fillna(-23333,inplace=True)
                # if mn >= 0:
                #     if mx < 255:
                #         props[col] = props[col].astype(np.uint8)
                #     elif mx < 65535:
                #         props[col] = props[col].astype(np.uint16)
                #     elif mx < 4294967295:
                #         props[col] = props[col].astype(np.uint32)
                #     else:
                #         props[col] = props[col].astype(np.uint64)
                # else:
                    # if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    #     props[col] = props[col].astype(np.int8)
                if mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    props[col] = props[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    props[col] = props[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

def dump_aux(obj, name):
    folder = 'features/aux_pickles'
    if not os.path.exists(folder):
        os.makedirs(folder)
    pickle.dump(obj, open(os.path.join(folder, '%s.pickle' % name), 'wb'))

def read_aux(name):
    folder =  'features/aux_pickles'
    return pickle.load(open(os.path.join(folder, '%s.pickle' % name), 'rb'))


def aggregate_by_region(id_name, column='logerror', force_generate=False):
    folder = 'features/aux_pickles'
    region_dict_name = '%s_%s' % (column, id_name)
    pickle_path = os.path.join(folder, '%s.pickle' % region_dict_name)
    if not force_generate and os.path.exists(pickle_path):
        return pickle.load(open(pickle_path, 'rb'))

    train, prop = load_train_data()
    df = train.merge(prop, how='left', on='parcelid')
    df, _ = split_by_date(df)

    region_dict = (df[[id_name, column]].groupby(id_name)
        .agg(['max', 'min', 'std', 'mean', 'count']).to_dict())

    default_value_dict = (
        df[[column]].agg(['max', 'min', 'std', 'mean']).to_dict()[column])
    region_dict['default'] = default_value_dict
    dump_aux(region_dict, region_dict_name)
    return region_dict
