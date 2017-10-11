from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import gc
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

def load_transaction_data(year, data_folder='data/', force_read=False):
    """ Load transaction data.
        Returns:
            train_df
    """
    train_file = 'train_2017'
    if year == 2016:
        train_file = 'train_2016_v2'

    train_data_pickle = data_folder + train_file + '_pickle'

    if not force_read and os.path.exists(train_data_pickle):
        train = pd.read_pickle(train_data_pickle)
    else:
        train = pd.read_csv(
            data_folder + train_file + '.csv', parse_dates=['transactiondate'])
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        train.to_pickle(train_data_pickle)
    return train

def load_properties_data_raw(year, data_folder='data/', force_read=False):
    """ Load raw properties data, for missing value calculation only now.
        Returns:
            properties_df
    """
    prop_data_pickle = '%sproperties_%s_raw_pickle' %(data_folder, year)

    if not force_read and os.path.exists(prop_data_pickle):
        prop = pd.read_pickle(prop_data_pickle)
    else:
        prop = pd.read_csv('%sproperties_%s.csv' %(data_folder, year))
        # prop = pd.read_csv('%sproperties_2017.csv' %data_folder)
        # # Should still use 2016's tax data to prevent leakage
        # if year == 2016:
        #     prop2016 = pd.read_csv('%sproperties_2016.csv' %data_folder)
        #     tax_columns = ['taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
        #         'landtaxvaluedollarcnt', 'taxamount' ,'assessmentyear',
        #         'taxdelinquencyflag' ,'taxdelinquencyyear']
        #     prop.drop(tax_columns, axis=1, inplace=True)
        #     tax2016 = prop2016[['parcelid', *tax_columns]]
        #     prop = prop.merge(tax2016, 'left', 'parcelid')
        #     del prop2016; del tax2016; gc.collect()
        for col in prop.columns:
            if prop[col].dtype == 'float64':
                prop[col] = prop[col].astype('float32')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        prop.to_pickle(prop_data_pickle)
    return prop

def load_properties_data(year, data_folder='data/', force_read=False):
    """ Load properties data.
        Returns:
            properties_df
    """
    infered_type = {
        "rawcensustractandblock": str,
        "censustractandblock": str,
        "propertycountylandusecode": str
    }
    prop_data_pickle = '%sproperties_%s_pickle' %(data_folder, year)

    if not force_read and os.path.exists(prop_data_pickle):
        prop = pd.read_pickle(prop_data_pickle)
    else:
        prop = pd.read_csv('%sproperties_%s.csv' %(data_folder, year), dtype=infered_type)
        # prop = pd.read_csv('%sproperties_2017.csv' %data_folder, dtype=infered_type)
        # # Should still use 2016's tax data to prevent leakage
        # if year == 2016:
        #     prop2016 = pd.read_csv('%sproperties_2016.csv' %data_folder)
        #     tax_columns = ['taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
        #         'landtaxvaluedollarcnt', 'taxamount' ,'assessmentyear',
        #         'taxdelinquencyflag' ,'taxdelinquencyyear']
        #     prop.drop(tax_columns, axis=1, inplace=True)
        #     tax2016 = prop2016[['parcelid', *tax_columns]]
        #     prop = prop.merge(tax2016, 'left', 'parcelid')
        #     del prop2016; del tax2016; gc.collect()
        # Fill missing geo data a little bit
        prop = preprocess_geo(prop)
        prop = preprocess_add_geo_features(prop)
        # Convert float64 to float32 to save memory
        for col in prop.columns:
            if prop[col].dtype == 'float64':
                prop[col] = prop[col].astype('float32')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        prop.to_pickle(prop_data_pickle)
    return prop


def load_properties_data_preprocessed(year, data_folder='data/', force_read=False):
    """ Load properties data and do some simple preprocess, mainly on boolean
        and categorical data.
        Returns:
            properties_df_preprocessed
    """
    prop_preprocessed_pickle_path = '%sproperties_%s_pickle_preprocessed' %(data_folder, year)
    if not force_read and os.path.exists(prop_preprocessed_pickle_path):
        prop_preprocessed = pd.read_pickle(prop_preprocessed_pickle_path)
    else:
        prop_preprocessed = load_properties_data(year, data_folder, force_read)

        # Preprocessing some columns so that all columns are numbers/booleans and has concrete meanings
        # boolean columns
        prop_preprocessed['fireplaceflag'] = feature_clean.fireplacecnt(prop_preprocessed)
        prop_preprocessed['hashottuborspa'] = feature_clean.hashottuborspa(prop_preprocessed)
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
        # prop_preprocessed['propertycountylandusecode'] = (
        #     labelEncoder.fit_transform(prop_preprocessed['propertycountylandusecode'].astype(str)))
        prop_preprocessed['propertyzoningdesc'] = (
            labelEncoder.fit_transform(prop_preprocessed['propertyzoningdesc'].astype(str)))

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        prop_preprocessed.to_pickle(prop_preprocessed_pickle_path)

    return prop_preprocessed


def load_properties_data_cleaned(year, data_folder='data/', force_read=False):
    """ Load properties data and clean data.
        Returns:
            properties_df_cleaned
    """
    prop_cleaned_pickle_path = '%sproperties_%d_pickle_cleaned' %(data_folder, year)
    if not force_read and os.path.exists(prop_cleaned_pickle_path):
        prop_cleaned = pd.read_pickle(prop_cleaned_pickle_path)
    else:
        # (name, function)
        functions = [o for o in inspect.getmembers(feature_clean) if inspect.isfunction(o[1])]
        # convert to a dictionary
        functions_dict = dict(functions)
        prop = load_properties_data(year, data_folder, force_read)
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

# # Load the reduced size properties data
# def load_properties_data_minimize(data_folder='data/', force_read=False):
#     """ Load properties data.
#         Returns:
#             properties_df
#     """
#     prop_data_pickle = '%s%s_pickle_mini' %(data_folder, source_data)

#     if not force_read and os.path.exists(prop_data_pickle):
#         prop = pd.read_pickle(prop_data_pickle)
#     else:
#         prop = pd.read_csv('%s%s.csv' %(data_folder, source_data))
#         # Should still use 2016's tax data to prevent leakage
#         if source_data == 'properties_2017':
#             prop2016 = pd.read_csv('%sproperties_2016.csv' %data_folder)
#             tax_columns = ['taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
#                 'landtaxvaluedollarcnt', 'taxamount' ,'assessmentyear',
#                 'taxdelinquencyflag' ,'taxdelinquencyyear']
#             prop.drop(tax_columns, axis=1, inplace=True)
#             tax2016 = prop2016[['parcelid', *tax_columns]]
#             prop = prop.merge(tax2016, 'left', 'parcelid')
#             del prop2016; del tax2016; gc.collect()
#         # Fill missing geo data a little bit
#         prop = preprocess_geo(prop)
#         prop = preprocess_add_geo_features(prop)
#         # Convert float64 to float32 to save memory
#         prop, na_list = reduce_mem_usage(prop)
#         prop.to_pickle(prop_data_pickle)
#     return prop

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
    test = sample.rename(columns={'ParcelId': 'parcelid'})
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


def add_date_features(df):
    print('Converting transaction features...')
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    df['transaction_year'] = df['transactiondate'].dt.year
    df['transaction_month'] = df['transactiondate'].dt.month
    # df['transaction_day'] = df['transactiondate'].dt.day
    df['transaction_quarter'] = df['transactiondate'].dt.quarter
    df['sin_dateofyear'] = np.sin(df['transactiondate'].dt.dayofyear / 365 * 2 * 3.1416)
    df['cos_dateofyear'] = np.cos(df['transactiondate'].dt.dayofyear / 365 * 2 * 3.1416)

    return df

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
    return (df.drop(non_feature_columns, axis=1), df[non_feature_columns])

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
    print('Object saved to %s/%s.pickle' % (folder, name))

def read_aux(name):
    folder =  'features/aux_pickles'
    print('Loading from %s/%s.pickle' % (folder, name))
    path = os.path.join(folder, '%s.pickle' % name)
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        print('File not exists')
        return None


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

##### Geo cleaning functions

def preprocess_add_geo_features(df):
    print('Process add Geo Features...')
    geodf = geo_everything(df)
    return pd.concat([df.drop(['regionidcity', 'regionidcounty', 'regionidzip', 'rawcensustractandblock', 'censustractandblock', 'propertycountylandusecode'], axis=1), geodf], axis=1)


def geo_everything(df):
    # split the geo part
    geocolumns = ['parcelid', 'latitude', 'longitude'
        , 'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc'
        , 'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip'
        , 'censustractandblock', 'rawcensustractandblock', 'fips']
    geoprop = df.loc[:, geocolumns]
    parcelid = df.parcelid

    geoprop.dropna(axis=0, subset=['latitude', 'longitude'], inplace=True)
    geoprop.loc[:, 'latitude'] = geoprop.loc[:, 'latitude'] / 1e6
    geoprop.loc[:, 'longitude'] = geoprop.loc[:, 'longitude'] / 1e6

    # fill the nan
    fillna_knn_inplace(df=geoprop,
                       base=['latitude', 'longitude'],
                       target='regionidcity', fraction=0.15)
    fillna_knn_inplace(df=geoprop,
                       base=['latitude', 'longitude'],
                       target='regionidzip', fraction=0.15)
    geoprop = create_category(geoprop, 'propertycountylandusecode')
    fillna_knn_inplace(df=geoprop,
                       base=['latitude', 'longitude'],
                       target='propertycountylandusecode', fraction=0.30)
    geoprop = add_census_block_feature(geoprop)

    # combine back
    geoprop = pd.merge(pd.DataFrame(parcelid), geoprop, how='outer', on='parcelid')

    # First liners are modified/cleaned features. Remove them from single feature functions.
    # Second line are new generated features.
    newfeatures = ['regionidcity', 'regionidcounty', 'regionidzip',
                   'propertycountylandusecode', 'fips_census_1', 'block_1', 'fips_census_block']
    return geoprop[newfeatures].astype(float)


# TODO: replace this by labelencoder
def create_category(df, name):
    df[name]=pd.factorize(df[name])[0]
    df.loc[df[name]<0, name] = np.nan
    return df

def add_census_block_feature(df):
    df2 = df.rawcensustractandblock.str.extract('(?P<fips_census_1>\d{9})\.(?P<block_1>\d*)')
    df2['fips_census_block'] = df2['fips_census_1'] + df2['block_1']
    return pd.concat([df, df2], axis=1)


from sklearn import neighbors
from sklearn.preprocessing import OneHotEncoder


def fillna_knn_inplace(df, base, target, fraction=1, threshold=10):
    assert isinstance(base, list) or isinstance(base, np.ndarray) and isinstance(target, str)
    whole = [target] + base  # [regionidcity, lat, lon]

    miss = df[target].isnull()
    notmiss = ~miss
    nummiss = miss.sum()

    enc = OneHotEncoder()
    X_target = df.loc[notmiss, whole].sample(frac=fraction)

    enc.fit(X_target[target].unique().reshape((-1, 1)))

    Y = enc.transform(X_target[target].values.reshape((-1, 1))).toarray()
    X = X_target[base]

    print('fitting')
    n_neighbors = 5
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X, Y)

    print('the shape of active features: ', enc.active_features_.shape)

    print('perdicting')
    Z = clf.predict(df.loc[miss, base])

    numunperdicted = Z[:, 0].sum()
    if numunperdicted / nummiss * 100 < threshold:
        print('writing result to df')
        df.loc[miss, target] = np.dot(Z, enc.active_features_)
        print('num of unperdictable data: ', numunperdicted)
    else:
        print('out of threshold: {}% > {}%'.format(numunperdicted / nummiss * 100, threshold))


# kernel density helper
from sklearn.neighbors import KDTree

def generate_pde_train(train, bandwidth):
    x = train['longitude'].values
    y = train['latitude'].values
    X = np.transpose(np.vstack([x, y]))
    print('Generating tree for bandwidth %s' % bandwidth)
    tree = KDTree(X, leaf_size=20)
    dump_aux(tree, 'pdf_tree_%s' % str(bandwidth))
    return tree


def generate_pde_test(test, bandwidth, force_generate=False):
    tree_name = 'pdf_tree_%s' % str(bandwidth)
    tree = read_aux(tree_name)
    if tree is None:
        train, prop = load_train_data()
        train = train.merge(prop, how='left', on='parcelid')
        train = train.dropna(subset=['longitude', 'latitude'])
        tree = generate_pde_train(train, bandwidth)
    test = test.dropna(subset=['longitude', 'latitude'])
    testidx = test.index
    x_test = test['longitude'].values
    y_test = test['latitude'].values
    X_test = np.transpose(np.vstack([x_test, y_test]))

    print('Generating PDE on test data for bandwidth %s' % bandwidth)
    parcelDensity = tree.kernel_density(X_test, h=bandwidth, kernel='gaussian', rtol=0.00001)
    print('Finished Generating')
    return pd.Series(parcelDensity, index=testidx)
