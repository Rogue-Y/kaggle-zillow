import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# import feature_eng.feature_eng as feature_eng

def remove_col(df, cols):
    df.drop(cols, axis=1, inplace=True)
    return df

def remove_outliers(df, llimit, ulimit):
    return df[(df['logerror'] >= llimit) & (df['logerror'] <= ulimit)]

# def remove_outliers(df, lpercentile, upercentile):
#     ulimit = np.percentile(df.logerror.values, upercentile)
#     llimit = np.percentile(df.logerror.values, lpercentile)
#     return df[(df['logerror'] >= llimit) & (df['logerror'] <= ulimit)]

def cat2num(df):
    labelEncoder = LabelEncoder()
    df['propertycountylandusecode'] = labelEncoder.fit_transform(df['propertycountylandusecode'].astype(str))
    df['propertyzoningdesc'] = labelEncoder.fit_transform(df['propertyzoningdesc'].astype(str))
    df['taxdelinquencyflag'] = df['taxdelinquencyflag'] == 'Y'
    df['fireplaceflag'] = df['fireplaceflag'] == True
    df['hashottuborspa'] = df['hashottuborspa'] == True
    # for c in df.dtypes[df.dtypes == object].index.values:
    #     df[c] = (df[c] == True)
    return df

def test(df, param='world'):
    print("hello, %s" %param)
    return df

def drop_low_ratio_columns(df):
    """ Drop feature with low non-nan ratio,
        current threshold is < 0.1 in properties dataframe.
        Exceptions are boolean columns: fireplaceflag, hashottuborspa,
        pooltypeid10, pooltypeid2, storytypeid
        Returns:
            df with some columns dropped
    """
    columns_to_drop = ['architecturalstyletypeid', 'basementsqft',
        'buildingclasstypeid', 'decktypeid', 'finishedfloor1squarefeet',
        'finishedsquarefeet6', 'finishedsquarefeet13', 'finishedsquarefeet15',
        'finishedsquarefeet50', 'poolsizesum', 'typeconstructiontypeid',
        'yardbuildingsqft17', 'yardbuildingsqft26', 'taxdelinquencyflag',
        'taxdelinquencyyear']
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df

def drop_high_corr_columns(df):
    """ Drop columns with high correlation with other columns
        Returns:
            df with some high correlation columns dropped
    """
    columns_to_drop = ['taxamount', 'taxvaluedollarcnt', 'calculatedbathnbr',
        'finishedsquarefeet12', 'fullbathcnt']


def drop_categorical_data(df):
    """
        Drop categorical columns.
    """
    columns_to_drop = ['airconditioningtypeid', 'fips', 'heatingorsystemtypeid',
        'propertylandusetypeid', 'propertycountylandusecode', 'propertyzoningdesc']
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df

def encode_categorical_data(df):
    """ Convert categorical data label to integers
        Returns:
            a encoded dataframe
    """
    labelEncoder = LabelEncoder()

    # fips count 90275
    df['fips'] = labelEncoder.fit_transform(df['fips'])

    # heatingorsystemtypeid count 56080
    df['heatingorsystemtypeid'] = labelEncoder.fit_transform(df['heatingorsystemtypeid'])

    # propertylandusetypeid
    df['propertylandusetypeid'] = labelEncoder.fit_transform(df['propertylandusetypeid'])

    land_use_code_threshold = 50000
    luc_vc = df['propertycountylandusecode'].value_counts()
    land_use_code = df['propertycountylandusecode'].mask(df['propertycountylandusecode'].map(luc_vc) < land_use_code_threshold, 'others')
    land_use_code = land_use_code.astype('str')
    df['propertycountylandusecode'] = labelEncoder.fit_transform(land_use_code)

    land_use_desc_threshold = 10000
    lud_vc = df['propertyzoningdesc'].value_counts()
    land_use_desc = df['propertyzoningdesc'].mask(df['propertyzoningdesc'].map(lud_vc) < land_use_desc_threshold, 'others')
    land_use_desc = land_use_desc.astype('str')
    df['propertyzoningdesc'] = labelEncoder.fit_transform(land_use_desc)

    return df

def one_hot_encode_categorical_data(df):
    """ Convert categorical data to one hot encode
        Returns:
            encoded dataframe
        columns to drop:
        ['airconditioningtypeid', 'fips', 'heatingorsystemtypeid',
        'propertycountylandusecode', 'propertyzoningdesc']
    """
    # Preparation
    labelEncoder = LabelEncoder()

    # airconditioningtypeid count 28781
    ac_dummies = pd.get_dummies(df['airconditioningtypeid'], prefix='airconditioningtypeid')

    # architecturalstyletypeid count 261
    #arch_style_dummies = pd.get_dummies(df['architecturalstyletypeid'], prefix='architecturalstyletypeid')

    # buildingclasstypeid count 16
    #building_class_dummies = pd.get_dummies(df['buildingclasstypeid'], prefix='buildingclasstypeid')

    # decktypeid count 658
    #deck_type_dummies = pd.get_dummies(df['decktypeid'], prefix='decktypeid')

    # fips count 90275
    fips_dummies = pd.get_dummies(df['fips'], prefix='fips')

    # heatingorsystemtypeid count 56080
    heating_dummies = pd.get_dummies(df['heatingorsystemtypeid'], prefix='heatingorsystemtypeid')

    # propertylandusetypeid
    land_use_id_dummies = pd.get_dummies(df['propertylandusetypeid'], prefix='propertylandusetypeid')

    # propertycountylandusecode count 90274
    # Replace low frequency count as 'others', and encode the values with integers
    land_use_code_threshold = 50000
    luc_vc = df['propertycountylandusecode'].value_counts()
    land_use_code = df['propertycountylandusecode'].mask(df['propertycountylandusecode'].map(luc_vc) < land_use_code_threshold, 'others')
    land_use_code = land_use_code.astype('str')
    land_use_code = labelEncoder.fit_transform(land_use_code)
    land_use_code_dummies = pd.get_dummies(land_use_code, prefix='propertycountylandusecode')

    # propertycountylandusecode count 58313
    # Replace low frequency count as 'others', and encode the values with integers
    land_use_desc_threshold = 10000
    lud_vc = df['propertyzoningdesc'].value_counts()
    land_use_desc = df['propertyzoningdesc'].mask(df['propertyzoningdesc'].map(lud_vc) < land_use_desc_threshold, 'others')
    land_use_desc = land_use_desc.astype('str')
    land_use_desc = labelEncoder.fit_transform(land_use_desc)
    land_use_desc_dummies = pd.get_dummies(land_use_desc, prefix='propertyzoningdesc')

    # typeconstructiontypeid count 299
    #construct_mat_dummies = pd.get_dummies(df['typeconstructiontypeid'], prefix='typeconstructiontypeid')

    columns_to_drop = ['airconditioningtypeid', 'fips', 'heatingorsystemtypeid',
        'propertylandusetypeid', 'propertycountylandusecode', 'propertyzoningdesc']

    df.drop(columns_to_drop, axis=1, inplace=True)

    df_list = [df, ac_dummies, fips_dummies, heating_dummies, land_use_id_dummies,
        land_use_code_dummies, land_use_desc_dummies]

    # df_lists_low_ratio = [arch_style_dummies, building_class_dummies,
    #     deck_type_dummies, construct_mat_dummies]

    return pd.concat(df_list, axis=1, copy=False)

def clean_boolean_data(df):
    """ Fill boolean data (replace nan with False / 0), inplace change, do not
        need to drop columns.
        Returns:
            the cleaned dataframe
        columns:
            ['fireplaceflag', 'hashottuborspa', 'pooltypeid10',
                'pooltypeid2', 'pooltypeid7', 'taxdelinquencyflag']
    """

    # fireplaceflag true count 222
    df['fireplaceflag'].fillna(False, inplace=True)

    # hashottuborspa true count 2365
    df['hashottuborspa'].fillna(False, inplace=True)

    # pooltypeid10 1 count 1161
    df['pooltypeid10'].fillna(0, inplace=True)

    # pooltypeid2 1 count 1204
    df['pooltypeid2'].fillna(0, inplace=True)

    # pooltypeid7 1 count 16697
    df['pooltypeid7'].fillna(0, inplace=True)

    # # taxdelinquencyflag Y count 1783
    # df['taxdelinquencyflag'] = df['taxdelinquencyflag'] == 'Y'

    # storytypeid (all 7, basically it's a isBasement flag)
    df['storytypeid'] = df['storytypeid'] == 7

    return df

# def clean_geo_data(df, lat_bin_num=10, lon_bin_num=10):
#     """ Need to think about how to deal with geo data, do nothing here and
#         drop the columns in column drop method for now.
#         Returns:
#             dataframe with geo feature added
#         columns (maybe to drop):
#             ['latitude', 'longitude', 'rawcensustractandblock',
#             'censustractandblock', 'regionidcounty', 'regionidcity',
#             'regionidzip', 'regionidneighborhood']
#     """
#     geo_columns = ['latitude', 'longitude', 'rawcensustractandblock',
#     'censustractandblock', 'regionidcounty', 'regionidcity',
#     'regionidzip', 'regionidneighborhood']
#
#     # put latitude and longitude into bins
#     lat_bins = pd.cut(df['latitude'], lat_bin_num, labels=False)
#     lat_bin_dummies = pd.get_dummies(lat_bins, prefix="lat_bin")
#     lon_bins = pd.cut(df['longitude'], lon_bin_num, labels=False)
#     lon_bin_dummies = pd.get_dummies(lon_bins, prefix="lon_bin")
#
#     # get dummies for 3 counties
#     county_dummies = pd.get_dummies(df['regionidcounty'], prefix='county')
#
#     df_list = [df, lat_bin_dummies, lon_bin_dummies, county_dummies]
#
#     df = pd.concat(df_list, axis=1, input=True)
#
#     # Cross latitude and longitude bins, and drop the single bins, as only
#     # latitude or longitude does not make much sense.
#     df = feature_eng.cross_features(df, 'lat_bin', 'lon_bin', '-');
#
#     lat_bin_cols = [col for col in df.columns if 'lat_bin' in col and '-' not in col]
#     lon_bin_cols = [col for col in df.columns if 'lon_bin' in col and '-' not in col]
#     df.drop(lat_bin_cols + lon_bin_cols, axis=1, inplace=True)
#     return df


# def drop_columns(df):
#     """ Drop un-used columns
#         Returns:
#             a copy of dataframe with un-used columns dropped
#     """
#     cat_columns = ['airconditioningtypeid', 'architecturalstyletypeid',
#     'buildingclasstypeid', 'decktypeid', 'fips', 'heatingorsystemtypeid',
#     'propertycountylandusecode', 'propertyzoningdesc', 'storytypeid',
#     'typeconstructiontypeid']
#     geo_columns = ['latitude', 'longitude', 'rawcensustractandblock',
#     'censustractandblock', 'regionidcounty', 'regionidcity',
#     'regionidzip', 'regionidneighborhood']
#     # Training data only columns
#     # train_columns = ['transactiondate']
#     # id_column = ['parcelid']
#
#     columns_to_drop = cat_columns + geo_columns
#     return df.drop(columns_to_drop, axis=1)

def drop_id_column(df):
    df.drop('parcelid', axis=1, inplace=True)
    return df

def drop_training_only_column(df):
    """ Drop the columns that is only available in training data,
        namely, parcelid and transactiondate
        Returns:
            the dataframe with training only columns dropped.
    """
    df.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    return df

def clean_strange_value(df, value=0):
    """ Violently fill all nan and inf as value, default to 0.
        Returns:
            a copy of df with nan and inf filled as value.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(value, inplace=True)
    return df
