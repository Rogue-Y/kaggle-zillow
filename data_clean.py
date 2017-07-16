import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fill_scalar(df, value):
    return df.fillna(value)

def remove_col(df, cols):
    return df.drop(cols, axis=1)

def remove_outliers(df, lpercentile, upercentile):
    ulimit = np.percentile(df.logerror.values, upercentile)
    llimit = np.percentile(df.logerror.values, lpercentile)
    return df[(df['logerror'] >= llimit) & (df['logerror'] <= ulimit)]

def cat2num(df):
    for c in df.dtypes[df.dtypes == object].index.values:
        df[c] = (df[c] == True)

def test():
    print("hello, world!")

def clean_categorical_data(df):
    """ Clean categorical data
        Returns:
            a copy of cleaned dataframe
        columns to drop:
        ['airconditioningtypeid', 'architecturalstyletypeid',
        'buildingclasstypeid', 'decktypeid', 'fips', 'heatingorsystemtypeid',
        'propertycountylandusecode', 'propertyzoningdesc', 'storytypeid',
        'typeconstructiontypeid']
    """
    # Preparation
    labelEncoder = LabelEncoder()

    # Make a copy so not mess up with the original df
    df = df.copy()

    # airconditioningtypeid count 28781
    ac_dummies = pd.get_dummies(df['airconditioningtypeid'], prefix='airconditioningtypeid')

    # architecturalstyletypeid count 261
    arch_style_dummies = pd.get_dummies(df['architecturalstyletypeid'], prefix='architecturalstyletypeid')

    # buildingclasstypeid count 16
    building_class_dummies = pd.get_dummies(df['buildingclasstypeid'], prefix='buildingclasstypeid')

    # decktypeid count 658
    deck_type_dummies = pd.get_dummies(df['decktypeid'], prefix='decktypeid')

    # fips count 90275
    fips_dummies = pd.get_dummies(df['fips'], prefix='fips')

    # heatingorsystemtypeid count 56080
    heating_dummies = pd.get_dummies(df['heatingorsystemtypeid'], prefix='heatingorsystemtypeid')

    # propertycountylandusecode count 90274
    # Replace low frequency count as 'others', and encode the values with integers
    land_use_code_threshold = 100
    luc_vc = df['propertycountylandusecode'].value_counts()
    land_use_code = df['propertycountylandusecode'].mask(df['propertycountylandusecode'].map(luc_vc) < land_use_code_threshold, 'others')
    land_use_code = land_use_code.astype('str')
    land_use_code = labelEncoder.fit_transform(land_use_code)
    land_use_code_dummies = pd.get_dummies(land_use_code, prefix='propertycountylandusecode')

    # propertycountylandusecode count 58313
    # Replace low frequency count as 'others', and encode the values with integers
    land_use_desc_threshold = 500
    lud_vc = df['propertyzoningdesc'].value_counts()
    land_use_desc = df['propertyzoningdesc'].mask(df['propertyzoningdesc'].map(lud_vc) < land_use_desc_threshold, 'others')
    land_use_desc = land_use_desc.astype('str')
    land_use_desc = labelEncoder.fit_transform(land_use_desc)
    land_use_desc_dummies = pd.get_dummies(land_use_desc, prefix='propertyzoningdesc')

    # storytypeid count 43
    story_type_dummies = pd.get_dummies(df['storytypeid'], prefix='storytypeid')

    # typeconstructiontypeid count 299
    construct_mat_dummies = pd.get_dummies(df['typeconstructiontypeid'], prefix='typeconstructiontypeid')

    df_lists = [df, ac_dummies, arch_style_dummies, building_class_dummies,
        deck_type_dummies, fips_dummies, heating_dummies, land_use_code_dummies,
        land_use_desc_dummies, story_type_dummies, construct_mat_dummies]

    return pd.concat(df_lists, axis=1)

def clean_boolean_data(df):
    """ Fill boolean data (replace nan with False / 0), inplace change, do not
        need to drop columns.
        Returns:
            a copy of cleaned dataframe
        columns:
            ['fireplaceflag', 'hashottuborspa', 'pooltypeid10',
                'pooltypeid2', 'pooltypeid7', 'taxdelinquencyflag']
    """

    # Make a copy so not mess up with the original df
    df = df.copy()

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

    # taxdelinquencyflag Y count 1783
    df['taxdelinquencyflag'] = df['taxdelinquencyflag'] == 'Y'

    return df

def clean_geo_data(df):
    """ Need to think about how to deal with geo data, do nothing here and
        drop the columns in column drop method for now.
        Returns:
            a copy of cleaned dataframe
        columns (maybe to drop):
            ['latitude', 'longitude', 'rawcensustractandblock',
            'censustractandblock', 'regionidcounty', 'regionidcity',
            'regionidzip', 'regionidneighborhood']
    """
    pass;

def drop_columns(df, isTrain):
    """ Drop un-used columns
        Params:
            isTrain: if df is training dataset
        Returns:
            a copy of dataframe with un-used columns dropped
    """
    cat_columns = ['airconditioningtypeid', 'architecturalstyletypeid',
    'buildingclasstypeid', 'decktypeid', 'fips', 'heatingorsystemtypeid',
    'propertycountylandusecode', 'propertyzoningdesc', 'storytypeid',
    'typeconstructiontypeid']
    geo_columns = ['latitude', 'longitude', 'rawcensustractandblock',
    'censustractandblock', 'regionidcounty', 'regionidcity',
    'regionidzip', 'regionidneighborhood']
    # Training data only columns
    train_columns = ['transactiondate']
    id_column = ['parcelid']

    columns_to_drop = cat_columns + geo_columns + id_column
    if isTrain:
        columns_to_drop = columns_to_drop + train_columns
    return df.drop(columns_to_drop, axis=1)
