# feature engineering
# TODO(hzn):
#   1. split out the columns in the original prop dataframe as single feature too;
#   2. add transformer for features, like fill nan, inf; normalization;
from collections import defaultdict

import pandas as pd
from .utils import *
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()

# methods from feature clean
def has_fireplace(df):
    return df['fireplacecnt'].notnull()

def is_garagetotalsqft_zero(df):
    return df['garagetotalsqft'] == 0

# Some garage has 0 carcnt but non-zero sqft
def has_partial_garagecarcnt(df):
    return (df['garagetotalsqft'] > 0) & (df['garagecarcnt'] == 0)

# 4 has 39877 count, 5 has only 588
def is_unitcnt_gt_four(df):
    return df['unitcnt'] > 4

# This could be replicate of the above one, since we fill 0 for the above one.
def has_shed_in_yard(df):
    return df['yardbuildingsqft26'].notnull()

def is_numberofstories_gt_three(df):
    return df['numberofstories'] > 3

def is_assessmentyear_2015(df):
    return df['assessmentyear'] == 2015

def is_tax_assessed(df):
    return df['assessmentyear'].notnull()

# Also try buckets for this one.
def is_taxdelinquencyyear_before_2014(df):
    return df['taxdelinquencyyear'] < 2014

# The difference between the sum of structure and land tax with total tax
def tax_difference(df):
    return (df['structuretaxvaluedollarcnt'] + df['landtaxvaluedollarcnt']
        - df['taxvaluedollarcnt'])

def has_construction_type(df):
    return df['typeconstructiontypeid'].notnull()

def is_roomcnt_zero(df):
    return df['roomcnt'] == 0



def missing_value_count(df):
    return df.isnull().sum(axis=1)

def missing_value_one_hot(df):
    missing_value_one_hot = pd.DataFrame()
    for col in df.columns:
        missing_value_one_hot[col+'_missing'] = df[col].isnull()
    return missing_value_one_hot

def building_age(df):
    return 2017 - df['yearbuilt']

def built_before_year(df, year=1900):
    return df['yearbuilt'] <= year

# TODO(hzn): investigate data leakage in these new features
# From https://www.kaggle.com/nikunjm88/creating-additional-features
# and https://www.kaggle.com/lauracozma/eda-data-cleaning-feature-engineering
def error_rate_calculated_finished_living_sqft(df):
    # error in calculation of the finished living area of home
    # this 2 values has correlation 1
    return ratio_helper(
        df['calculatedfinishedsquarefeet'],
        df['finishedsquarefeet12'])

def error_rate_first_floor_living_sqft(df):
    return ratio_helper(
        df['finishedsquarefeet50'],
        df['finishedfloor1squarefeet'])

def error_rate_bathroom(df):
    return ratio_helper(
        df['calculatedbathnbr'],
        df['bathroomcnt'])

def error_rate_count_bathroom(df):
    return ratio_helper(
        df['threequarterbathnbr'] + df['fullbathcnt'],
        df['bathroomcnt'])

def ratio_living_area(df):
    #proportion of living area
    return ratio_helper(
        df['calculatedfinishedsquarefeet'],
        df['lotsizesquarefeet'])

def ratio_living_area_2(df):
    return ratio_helper(
        df['finishedsquarefeet12'],
        df['finishedsquarefeet15'])

def ratio_bedroom_bathroom(df):
    return ratio_helper(
        df['bedroomcnt'],
        df['bathroomcnt'])

def ratio_basement(df):
    return ratio_helper(
        df['basementsqft'],
        df['finishedsquarefeet12'])

def ratio_pool_yard(df):
    return ratio_helper(
        df['poolsizesum'],
        df['yardbuildingsqft17'])

def ratio_pool_shed(df):
    return ratio_helper(
        df['yardbuildingsqft26'],
        df['yardbuildingsqft17'])

def ratio_floor_shape(df):
    return ratio_helper(
        df['finishedsquarefeet13'],
        df['calculatedfinishedsquarefeet'])

def ratio_fireplace(df):
    return ratio_helper(
        df['fireplacecnt'],
        df['finishedsquarefeet15'])

def extra_space(df):
    #Amout of extra space
    return df['lotsizesquarefeet'] - df['calculatedfinishedsquarefeet']

def total_rooms(df):
    #Total number of rooms
    return df['bathroomcnt'] + df['bedroomcnt']

def average_room_size(df):
    #Average room size
    total = total_rooms(df)
    # To deal with inf and nan (when both denominator and nominator is 0),
    # when total is zero, default it to 1.
    return ratio_helper(df['calculatedfinishedsquarefeet'], total)

def clipped_average_room_size(df):
    feature = average_room_size(df)
    return clip_create_outlier_bool(feature, 'average_room_size', 0, 0.99)

# # roomcnt is mostly 0
# def average_room_size_2(df):
#     return df['calculatedfinishedsquarefeet']/df['roomcnt']

def average_bathroom_size(df):
    return ratio_helper(
        df['finishedsquarefeet12'],
        df['bathroomcnt'])

def average_bedroom_size(df):
    return ratio_helper(
        df['finishedsquarefeet12'],
        df['bedroomcnt'])

# poolcnt all 1
# def average_pool_size(df):
#     return df['poolsizesum']/df['poolcnt']

def extra_rooms(df):
    # Number of Extra rooms
    # most roomcnt are 0
    total = total_rooms(df)
    return total - df['roomcnt']

def boolean_has_garage_pool_and_ac(df):
    #Does property have a garage, pool or hot tub and AC?
    return (df['garagecarcnt']>0) & (df['poolcnt']>0) & (df['airconditioningtypeid']!=5)

# Tax related
def ratio_tax_value_to_structure_value(df):
    #Ratio of the total value to structure
    return df['taxvaluedollarcnt']/df['structuretaxvaluedollarcnt']

def ratio_tax_value_to_land_tax_value(df):
    #Ratio of the total value to land area
    return ratio_helper(
        df['taxvaluedollarcnt'],
        df['landtaxvaluedollarcnt'])

def ratio_structure_tax_value_to_land_tax_value(df):
    #Ratio of the built structure value to land area
    return ratio_helper(
        df['structuretaxvaluedollarcnt'],
        df['landtaxvaluedollarcnt'])

def ratio_tax(df):
    #Ratio of tax of property over parcel
    return ratio_helper(
        df['taxamount'],
        df['taxvaluedollarcnt'])

    # #TotalTaxScore
    # df['N-TaxScore'] = df['taxvaluedollarcnt']*df['taxamount']

# Geo features
def sum_lat_lon(df):
    return df["latitude"] + df["longitude"]

def multiply_lat_lon(df):
    return df["latitude"] * df["longitude"]

def round_multiply_lat_lon(df):
    return multiply_lat_lon(df).round(-4)

def round_lat(df):
    return df["latitude"].round(-4)

def round_lon(df):
    return df["longitude"].round(-4)

# TODO(hzn): add more aggregation to neighborhood/zip/city/county/lat-lon-block:
# https://pandas.pydata.org/pandas-docs/stable/api.html#id32
def geo_fips_census_block(df, columns=None):
    fips_census_block = pd.DataFrame()
    if columns is None:
        # Use default columns if col is None
        columns = [
            'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
            'calculatedfinishedsquarefeet', 'fullbathcnt', 'garagecarcnt',
            # 'garagetotalsqft', 'lotsizesquarefeet', 'numberofstories',
            # 'roomcnt',
            'unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt',
            'taxamount',
            'taxvaluedollarcnt']
    #Number of properties in the fips_census_block
    fips_census_block_count = df['fips_census_block'].value_counts().to_dict()
    fips_census_block['fips_census_block_count'] = df['fips_census_block'].map(fips_census_block_count)

    # stats of value estimate of properties grouped by fips_census_block
    fips_census_block_dict = (df[['fips_census_block', *columns]].groupby('fips_census_block')
        .agg(['max', 'min', 'std', 'mean']).to_dict())

    for col in columns:
        fips_census_block[col + '_fips_census_block_mean'] = df['fips_census_block'].map(fips_census_block_dict[(col, 'mean')])
        # when there's only one item in the group. the std method returns nan, fill it with 0.
        fips_census_block[col + '_fips_census_block_std'] = df['fips_census_block'].map(fips_census_block_dict[(col, 'std')]).fillna(0)
        # For those 2 ratios, when the denominator is zero, the nominator must be 0 too.
        fips_census_block[col + '_fips_census_block_mean_ratio'] = ratio_helper(
            df[col],
            fips_census_block[col + '_fips_census_block_mean'],
            1)
        fips_census_block[col + '_fips_census_block_mean_ratio'] = ratio_helper(
            (df[col] - fips_census_block[col + '_fips_census_block_mean']),
            fips_census_block[col + '_fips_census_block_std'],
            1)

    # For the parcels where the fips_census_block is unknown(nan in origianl dataset,
    # denoted by 0 after fillna), fill all values with zero.
    fips_census_block[df['fips_census_block'] == 0] = 0

    return fips_census_block

def geo_fips_census_1(df, columns=None):
    fips_census_1 = pd.DataFrame()
    if columns is None:
        # Use default columns if col is None
        columns = [
            'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
            'calculatedfinishedsquarefeet', 'fullbathcnt', 'garagecarcnt',
            # 'garagetotalsqft', 'lotsizesquarefeet', 'numberofstories',
            # 'roomcnt',
            'unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt',
            'taxamount',
            'taxvaluedollarcnt']
    #Number of properties in the fips_census_1
    fips_census_1_count = df['fips_census_1'].value_counts().to_dict()
    fips_census_1['fips_census_1_count'] = df['fips_census_1'].map(fips_census_1_count)

    # stats of value estimate of properties grouped by fips_census_1
    fips_census_1_dict = (df[['fips_census_1', *columns]].groupby('fips_census_1')
        .agg(['max', 'min', 'std', 'mean']).to_dict())

    for col in columns:
        fips_census_1[col + '_fips_census_1_mean'] = df['fips_census_1'].map(fips_census_1_dict[(col, 'mean')])
        # when there's only one item in the group. the std method returns nan, fill it with 0.
        fips_census_1[col + '_fips_census_1_std'] = df['fips_census_1'].map(fips_census_1_dict[(col, 'std')]).fillna(0)
        # For those 2 ratios, when the denominator is zero, the nominator must be 0 too.
        fips_census_1[col + '_fips_census_1_mean_ratio'] = ratio_helper(
            df[col],
            fips_census_1[col + '_fips_census_1_mean'],
            1)
        fips_census_1[col + '_fips_census_1_mean_ratio'] = ratio_helper(
            (df[col] - fips_census_1[col + '_fips_census_1_mean']),
            fips_census_1[col + '_fips_census_1_std'],
            1)

    # For the parcels where the fips_census_1 is unknown(nan in origianl dataset,
    # denoted by 0 after fillna), fill all values with zero.
    fips_census_1[df['fips_census_1'] == 0] = 0

    return fips_census_1

def geo_neighborhood(df, columns=None):
    neighborhood = pd.DataFrame()
    if columns is None:
        # Use default columns if col is None
        columns = [
            'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
            'calculatedfinishedsquarefeet', 'fullbathcnt', 'garagecarcnt',
            # 'garagetotalsqft', 'lotsizesquarefeet', 'numberofstories',
            # 'roomcnt',
            'unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt',
            'taxamount',
            'taxvaluedollarcnt']
    #Number of properties in the neighborhood
    neighborhood_count = df['regionidneighborhood'].value_counts().to_dict()
    neighborhood['neighborhood_count'] = df['regionidneighborhood'].map(neighborhood_count)

    # stats of value estimate of properties grouped by neighborhood
    neighborhood_dict = (df[['regionidneighborhood', *columns]].groupby('regionidneighborhood')
        .agg(['max', 'min', 'std', 'mean']).to_dict())

    for col in columns:
        neighborhood[col + '_neighborhood_mean'] = df['regionidneighborhood'].map(neighborhood_dict[(col, 'mean')])
        # when there's only one item in the group. the std method returns nan, fill it with 0.
        neighborhood[col + '_neighborhood_std'] = df['regionidneighborhood'].map(neighborhood_dict[(col, 'std')]).fillna(0)
        # For those 2 ratios, when the denominator is zero, the nominator must be 0 too.
        neighborhood[col + '_neighborhood_mean_ratio'] = ratio_helper(
            df[col],
            neighborhood[col + '_neighborhood_mean'],
            1)
        neighborhood[col + '_neighborhood_mean_ratio'] = ratio_helper(
            (df[col] - neighborhood[col + '_neighborhood_mean']),
            neighborhood[col + '_neighborhood_std'],
            1)
        # neighborhood.drop(col + '_neighborhood_mean', axis=1, inplace=True)
        # neighborhood.drop(col + '_neighborhood_std', axis=1, inplace=True)
        # neighborhood[col + '_neighborhood_max'] = df['regionidneighborhood'].map(neighborhood_dict[(col, 'max')])
        # neighborhood[col + '_neighborhood_min'] = df['regionidneighborhood'].map(neighborhood_dict[(col, 'min')])

    # For the parcels where the neighborhood_id is unknown(nan in origianl dataset,
    # denoted by 0 after fillna), fill all values with zero.
    neighborhood[df['regionidneighborhood'] == 0] = 0

    return neighborhood

def geo_city(df, columns=None):
    city = pd.DataFrame()
    if columns is None:
        # Use default columns if col is None
        columns = [
            'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
            'calculatedfinishedsquarefeet', 'fullbathcnt', 'garagecarcnt',
            # 'garagetotalsqft', 'lotsizesquarefeet', 'numberofstories',
            # 'roomcnt',
            'unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt',
            'taxamount',
            'taxvaluedollarcnt']
    #Number of properties in the city
    city_count = df['regionidcity'].value_counts().to_dict()
    city['city_count'] = df['regionidcity'].map(city_count)

    # stats of value estimate of properties grouped by city
    city_dict = (df[['regionidcity', *columns]].groupby('regionidcity')
        .agg(['max', 'min', 'std', 'mean']).to_dict())

    for col in columns:
        city[col + '_city_mean'] = df['regionidcity'].map(city_dict[(col, 'mean')])
        # when there's only one item in the group. the std method returns nan, fill it with 0.
        city[col + '_city_std'] = df['regionidcity'].map(city_dict[(col, 'std')]).fillna(0)
        city[col + '_city_mean_ratio'] = ratio_helper(
            df[col],
            city[col + '_city_mean'],
            1)
        city[col + '_city_std_ratio'] = ratio_helper(
            (df[col] - city[col + '_city_mean']),
            city[col + '_city_std'],
            1)
        # city[col + '_city_max'] = df['regionidcity'].map(city_dict[(col, 'max')])
        # city[col + '_city_min'] = df['regionidcity'].map(city_dict[(col, 'min')])

    city[df['regionidcity'] == 0] = 0

    return city

def geo_zip(df, columns=None):
    zip = pd.DataFrame()
    if columns is None:
        # Use default columns if col is None
        columns = [
            'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
            'calculatedfinishedsquarefeet', 'fullbathcnt', 'garagecarcnt',
            # 'garagetotalsqft', 'lotsizesquarefeet', 'numberofstories',
            # 'roomcnt',
            'unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt',
            'taxamount',
            'taxvaluedollarcnt']
    #Number of properties in the zip
    zip_count = df['regionidzip'].value_counts().to_dict()
    zip['zip_count'] = df['regionidzip'].map(zip_count)

    # stats of value estimate of properties grouped by zip
    zip_dict = (df[['regionidzip', *columns]].groupby('regionidzip')
        .agg(['max', 'min', 'std', 'mean']).to_dict())

    for col in columns:
        zip[col + '_zip_mean'] = df['regionidzip'].map(zip_dict[(col, 'mean')])
        # when there's only one item in the group. the std method returns nan, fill it with 0.
        zip[col + '_zip_std'] = df['regionidzip'].map(zip_dict[(col, 'std')]).fillna(0)
        zip[col + '_zip_mean_ratio'] = ratio_helper(
            df[col],
            zip[col + '_zip_mean'],
            1)
        zip[col + '_zip_std_ratio'] = ratio_helper(
            (df[col] - zip[col + '_zip_mean']),
            zip[col + '_zip_std'],
            1)
        # zip[col + '_zip_max'] = df['regionidzip'].map(zip_dict[(col, 'max')])
        # zip[col + '_zip_min'] = df['regionidzip'].map(zip_dict[(col, 'min')])

    zip[df['regionidzip'] == 0] = 0

    return zip

def category_geo_county_one_hot(df):
    return pd.get_dummies(df['regionidcounty'], prefix='regionidcounty')

def geo_county(df, columns=None):
    county = pd.DataFrame()
    if columns is None:
        # Use default columns if col is None
        columns = [
            'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
            'calculatedfinishedsquarefeet', 'fullbathcnt', 'garagecarcnt',
            # 'garagetotalsqft', 'lotsizesquarefeet', 'numberofstories',
            # 'roomcnt',
            'unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt',
            'taxamount',
            'taxvaluedollarcnt']
    #Number of properties in the county
    county_count = df['regionidcounty'].value_counts().to_dict()
    county['county_count'] = df['regionidcounty'].map(county_count)

    # stats of value estimate of properties grouped by county
    county_dict = (df[['regionidcounty', *columns]].groupby('regionidcounty')
        .agg(['max', 'min', 'std', 'mean']).to_dict())

    for col in columns:
        county[col + '_county_mean'] = df['regionidcounty'].map(county_dict[(col, 'mean')])
        # when there's only one item in the group. the std method returns nan, fill it with 0.
        county[col + '_county_std'] = df['regionidcounty'].map(county_dict[(col, 'std')]).fillna(0)
        county[col + '_county_mean_ratio'] = ratio_helper(
            df[col],
            county[col + '_county_mean'],
            1)
        county[col + '_county_std_ratio'] = ratio_helper(
            (df[col] - county[col + '_county_mean']),
            county[col + '_county_std'],
            1)
        # county[col + '_county_max'] = df['regionidcounty'].map(county_dict[(col, 'max')])
        # county[col + '_county_min'] = df['regionidcounty'].map(county_dict[(col, 'min')])

    county[df['regionidcounty'] == 0] = 0

    return county

def geo_lat_lon_block_helper(df):
    # Latitude, longitude blocks
    lat_bins = pd.cut(df['latitude'], 10, labels=False)
    lat_bins = labelEncoder.fit_transform(lat_bins)
    lon_bins = pd.cut(df['longitude'], 10, labels=False)
    lon_bins = labelEncoder.fit_transform(lon_bins)
    return pd.Series(lat_bins * 10 + lon_bins, name='lat_lon_block')

# def geo_lat_lon_block_tax_value(df):
#     lat_lon_block = pd.DataFrame()

#     blocks = geo_lat_lon_block_helper(df)

#     #Number of properties in the lat_lon_block
#     lat_lon_block_count = blocks.value_counts().to_dict()
#     lat_lon_block['lat_lon_block_count'] = blocks.map(lat_lon_block_count)

#     # stats of value estimate of properties grouped by lat_lon_block
#     lat_lon_block_tax_value_df = pd.DataFrame(
#         {'lat_lon_block': blocks, 'taxvaluedollarcnt': df['taxvaluedollarcnt']})
#     lat_lon_block_dict = (lat_lon_block_tax_value_df.groupby('lat_lon_block')
#         .agg(['max', 'min', 'std', 'mean'])['taxvaluedollarcnt'].to_dict())

#     lat_lon_block['lat_lon_block_value_mean'] = blocks.map(lat_lon_block_dict['mean'])
#     lat_lon_block['lat_lon_block_value_std'] = blocks.map(lat_lon_block_dict['std'])
#     lat_lon_block['lat_lon_block_value_max'] = blocks.map(lat_lon_block_dict['max'])
#     lat_lon_block['lat_lon_block_value_min'] = blocks.map(lat_lon_block_dict['min'])
#     lat_lon_block['lat_lon_block_value_range'] = lat_lon_block['lat_lon_block_value_max'] - lat_lon_block['lat_lon_block_value_min']

#     return lat_lon_block

def geo_lat_lon_block_features(df, columns=None):
    if columns is None:
        # Use default columns if col is None
        columns = [
            'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
            'calculatedfinishedsquarefeet', 'fullbathcnt', 'garagecarcnt',
            'garagetotalsqft', 'lotsizesquarefeet', 'numberofstories',
            # 'roomcnt',
            'unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt',
            'taxamount',
            'taxvaluedollarcnt']
    lat_lon_block = pd.DataFrame()

    blocks = geo_lat_lon_block_helper(df)
    lat_lon_block['lat_lon_block'] = blocks

    values = df.loc[:, columns]
    values['lat_lon_block'] = blocks

    # stats of value estimate of properties grouped by lat_lon_block
    lat_lon_block_dict = (values.groupby('lat_lon_block')
        .agg(['max', 'min', 'std', 'mean']).to_dict())

    for col in columns:
        lat_lon_block[col + '_lat_lon_block_mean'] = blocks.map(lat_lon_block_dict[(col, 'mean')])
        lat_lon_block[col + '_lat_lon_block_mean_ratio'] = ratio_helper(
            df[col],
            lat_lon_block[col + '_lat_lon_block_mean'],
            1)
        # when there's only one item in the group. the std method returns nan, fill it with 0.
        lat_lon_block[col + '_lat_lon_block_std'] = blocks.map(lat_lon_block_dict[(col, 'std')]).fillna(0)
        lat_lon_block[col + '_lat_lon_block_std_ratio'] = ratio_helper(
            (df[col] - lat_lon_block[col + '_lat_lon_block_mean']),
            lat_lon_block[col + '_lat_lon_block_std'],
            1)
        # lat_lon_block.drop(col + '_lat_lon_block_mean', axis=1, inplace=True)
        # lat_lon_block.drop(col + '_lat_lon_block_std', axis=1, inplace=True)
        # lat_lon_block[col + '_lat_lon_block_max'] = df['regionidlat_lon_block'].map(lat_lon_block_dict[(col, 'max')])
        # lat_lon_block[col + '_lat_lon_block_min'] = df['regionidlat_lon_block'].map(lat_lon_block_dict[(col, 'min')])

    return lat_lon_block


# def geo_city_structure_tax_value(df):
#     #Average structuretaxvaluedollarcnt by city
#     group = df.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
#     return df['regionidcity'].map(group)

# def deviation_from_avg_structure_tax_value(df):
#     #Deviation away from average
#     avg_structure_tax_value = geo_city_structure_tax_value(df)
#     return abs(df['structuretaxvaluedollarcnt'] - avg_structure_tax_value) / avg_structure_tax_value

# others
def boolean_has_ac(df):
    #Indicator whether it has AC or not
    return df['airconditioningtypeid']!=5

def boolean_has_heat(df):
    return df['heatingorsystemtypeid']!=13

# categories
def category_land_use_type_helper(df):
    #There's 25 different property uses - let's compress them down to 4 categories
    land_use_types = df['propertylandusetypeid'].replace({
        31 : "Mixed", 46 : "Other", 47 : "Mixed", 246 : "Mixed", 247 : "Mixed",
        248 : "Mixed", 260 : "Home", 261 : "Home", 262 : "Home", 263 : "Home",
        264 : "Home", 265 : "Home", 266 : "Home", 267 : "Home", 268 : "Home",
        269 : "Not Built", 270 : "Home", 271 : "Home", 273 : "Home",
        274 : "Other", 275 : "Home", 276 : "Home", 279 : "Home",
        290 : "Not Built", 291 : "Not Built" })
    return land_use_types.replace({"Mixed": 0, "Home": 1, "Not Built": 2, "Other": 3})

def category_land_use_type_encode(df):
    return pd.Series(labelEncoder.fit_transform(category_land_use_type_helper(df)))

def category_land_use_type_one_hot(df):
    return pd.get_dummies(category_land_use_type_helper(df), prefix='land_use_type')

# def category_ac_type_encode(df):
#     return pd.Series(labelEncoder.fit_transform(df['airconditioningtypeid']))

def category_ac_type_one_hot(df):
    return pd.get_dummies(df['airconditioningtypeid'], prefix='airconditioningtypeid')

def category_architecture_style_one_hot(df):
    return pd.get_dummies(df['architecturalstyletypeid'], prefix='architecturalstyletypeid')

def category_building_class_one_hot(df):
    return pd.get_dummies(df['buildingclasstypeid'], prefix='buildingclasstypeid')

# def category_fips_type_encode(df):
#     return pd.Series(labelEncoder.fit_transform(df['fips']))

def category_fips_type_one_hot(df):
    return pd.get_dummies(df['fips'], prefix='fips')

# def category_heating_type_encode(df):
#     return pd.Series(labelEncoder.fit_transform(df['heatingorsystemtypeid']))

def category_heating_type_one_hot(df):
    # Replace the non-popular types as others
    # TODO: try group with meanings instead of cut of counts
    tmp = df['heatingorsystemtypeid'].replace([20, 6, 18, 24, 12, 10, 1, 14, 21, 11, 19], 0)
    return pd.get_dummies(tmp, prefix='heatingorsystemtypeid')

def category_land_use_code_helper(df):
    land_use_code_threshold = 50000
    luc_vc = df['propertycountylandusecode'].value_counts()
    land_use_code = df['propertycountylandusecode'].mask(df['propertycountylandusecode'].map(luc_vc) < land_use_code_threshold, 'others')
    return land_use_code.astype('str')

def category_land_use_code_encode(df):
    return pd.Series(labelEncoder.fit_transform(category_land_use_code_helper(df)))

def category_land_use_code_one_hot(df):
    return pd.get_dummies(category_land_use_code_helper(df), prefix='propertycountylandusecode')

def category_land_use_desc_helper(df):
    land_use_desc_threshold = 10000
    lud_vc = df['propertyzoningdesc'].value_counts()
    land_use_desc = df['propertyzoningdesc'].mask(df['propertyzoningdesc'].map(lud_vc) < land_use_desc_threshold, 'others')
    return land_use_desc.astype('str')

def category_land_use_desc_encode(df):
    return pd.Series(labelEncoder.fit_transform(category_land_use_desc_helper(df)))

def category_land_use_desc_one_hot(df):
    return pd.get_dummies(category_land_use_desc_helper(df), prefix='propertyzoningdesc')

def category_construction_type_one_hot(df):
    # Replace the non-popular types as others (0 is used to fillna here)
    # TODO: try group with meanings instead of cut of counts
    tmp = df['typeconstructiontypeid'].replace([4, 10, 13, 11], 1)
    return pd.get_dummies(tmp, prefix='typeconstructiontypeid')

def category_tax_delinquency_year_one_hot(df):
    # Replace the non-popular types as others
    tmp = df['taxdelinquencyyear'].where(df['taxdelinquencyyear'] >= 2014, 0)
    return pd.get_dummies(tmp, prefix='taxdelinquencyyear')

#polnomials of the variable
def poly_2_structure_tax_value(df):
    return df["structuretaxvaluedollarcnt"] ** 2

def poly_3_structure_tax_value(df):
    return df["structuretaxvaluedollarcnt"] ** 3


########### Help functions #############
# replace the 0s in denominator with the given default_value value,
# if no default value is provided, used the minimum non-zero value of the
# denominator instead. This is because the data we are deal with are all
# non-nega
def ratio_helper(nominator, denominator, default_value=None):
    if default_value is None:
        default_value = denominator.iloc[denominator.nonzero()[0]].min()
    result = nominator / denominator
    return result.where(denominator != 0, nominator / default_value)

# def get_distance(p1, p2):
#     """ This use a rough calculation of distance, will overestimate the distance
#         for latitude around 30 degree
#         for more, see: http://www.movable-type.co.uk/scripts/latlong.html
#         the lat / lon in data set divided by 10^6 is degree, and need to be
#         converted to radian for distance calculation
#         the calculated distance is in km
#     """
#     # Radius of the earth
#     R = 6371;
#     pi = 3.14
#     scale = 10**6
#     diff_lat = (p1[0] - p2[0]) * pi / scale / 180
#     diff_lon = (p1[1] - p2[1]) * pi / scale / 180
#     return math.hypot(diff_lat, diff_lon) * R
#
# def add_nearby_houses(df, threshold=5):
#     """ Calulate how many nearby houses within threshold km
#         Returns:
#             a copy of df with new feature added.
#     """
#     rows = df.shape[0]
#     df['nearby_houses'] = np.zeros(rows)
#     df['nearby_houses'] = df['nearby_houses'].astype(int)
#     for i in range(rows):
#         nearby_houses_i = df.loc[i, 'nearby_houses']
#         for j in range(i, rows):
#             p1 = (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
#             p2 = (df.loc[j, 'latitude'], df.loc[j, 'longitude'])
#             if get_distance(p1, p2) < threshold:
#                 nearby_houses_i = nearby_houses_i + 1
#                 df.loc[j, 'nearby_houses'] = df.loc[j, 'nearby_houses'] + 1
#         df.loc[i, 'nearby_houses'] = nearby_houses_i
#     df.to_csv('data/add_nearby_houses.csv', index=True)
#     return df

def add_bins(df, column, quantile=False, bin_num=10, one_hot_encode=True):
    """ Put a features into bins based on value (use pd.cut) or
        quantile (use pd.qcut), and optionally convert the bins into
        one-hot encode columns.
        Params:
            column - column name
            bin_num - could be integer(evenly cut), or an array indicate boundaries
                (left exculded, right included),
                note that when cut by quantile, those boundaries needs to be percentiles
        Returns:
            df with bins columns(optionally one-hot encoded) added.
            bins column will be called column_bins
            one-hot encoded bins columns are prefix with column_bin
    """

    # TODO(hzn): think about if use cut on the whole data set will cause data
    # leakage
    cut_to_use = pd.qcut if quantile else pd.cut
    bins = cut_to_use(df[column], bin_num, labels=False) # labels = False marks bins as integers

    if not one_hot_encode:
        df[column+'_bins'] = bins
        return df

    bin_dummies = pd.get_dummies(bins, prefix=column+'_bin')
    return pd.concat([df, bin_dummies], axis=1, copy=False)

def add_year_tax_bins(df, year_bin_num=10, tax_bin_num=10):
    """ Put some features into bins.
        Returns:
            df with bins columns added.
        Columns:
            ['yearbuilt', 'taxamount']
    """
    df = add_bins(df, 'yearbuilt', True, year_bin_num, True)
    df = add_bins(df, 'taxamount', True, tax_bin_num, True)

    return df

# TODO(hzn): expand this method to more than 2 features if needed
def cross_features(df, feature1, feature2, cross_notation='*'):
    """ Cross feature1 and feature2
        Params:
            cross_notation: default to *, and we use * as a sign of crossed columns,
                And exclude those columns from being included in the cross list.
                user of this function can pass in other notations so that the
                crossed columns can still be included in the cross list, e.g.
                crossed latitude and longitude bins in data_clean.clean_geo_data().
        Returns:
            the dataframe with crossed columns added.
    """
    # columns are the same
    if feature1 == feature2:
        return df

    # All columns related to feature1 or feature2, '*' used to denote crossed
    # columns, they should not be included in the columns to cross
    columns_list_one = [col for col in df.columns if feature1 in col and '*' not in col]
    columns_list_two = [col for col in df.columns if feature2 in col and '*' not in col]

    for col_i in columns_list_one:
        for col_j in columns_list_two:
            # when cross_notation is '*', the resulting column name is col_i*col_j
            df[col_i+ cross_notation +col_j] = df[col_i] * df[col_j]

    return df

def feature_crossing(df, cross_list=[]):
    """ Cross a list of features
        Params:
            df - the dataframe
            cross_list - a list of feature tuples that needs to be crossed
        Returns:
            dataframe with crossed columns added.
    """
    for feature_tuple in cross_list:
        cross_features(df, *feature_tuple)

    return df

def target_region_feature(df, id_name, column='logerror', minthres=10):
    region_dict = aggregate_by_region(id_name, column)
    default_value_dict = region_dict['default']
    del region_dict['default']

    print(region_dict.keys())
    stats = [item[1] for item in region_dict.keys() if item[1] != 'count']
    print(stats)
    print('Total Group number', len(region_dict[(column, 'count')]))

    for k,v in region_dict[(column, 'count')].items():
        if v < minthres:
            for stat in stats:
                del region_dict[(column, stat)][k]

    print('Survived Group number', len(region_dict[(column, stats[0])]))
    newdf = pd.DataFrame()
    for stat in stats:
        newdf[column + '_' + id_name + '_' + stat] = df[id_name].map(region_dict[column, stat])
        newdf[column + '_' + id_name + '_' + stat].fillna(default_value_dict[stat])
    newdf[column + '_' + id_name + '_std_over_mean'] = newdf[column + '_' + id_name + '_std'] / newdf[column + '_' + id_name + '_mean']
    newdf[column + '_' + id_name + '_range'] = newdf[column + '_' + id_name + '_max'] - newdf[column + '_' + id_name + '_min']

    print(list(newdf))
    newdf = newdf.replace([np.inf, -np.inf], np.nan)
    newdf = newdf.fillna(0)
    print('Target: Fill NaN and Inf with 0')
    return newdf

def target_region_ratio(df):
    newdf = pd.DataFrame()
    column = 'logerror'
    stats = ['std', 'mean', 'max', 'min']
    cases = [('regionidneighborhood', 'regionidcity'), ('regionidneighborhood', 'regionidzip'), ('regionidneighborhood', 'regionidcounty')]
    for (r1, r2) in cases:
        dfr1 = target_region_feature(df, r1, column)
        dfr2 = target_region_feature(df, r2, column)
        for stat in stats:
            newdf[column + '_' + stat + '_' + 'ratio' + r1 + '_' + r2] = dfr1[column + '_' + r1 + '_' + stat] / dfr2[column + '_' + r2 + '_' + stat]
    return newdf

def clip_create_outlier_bool_helper(feature, name, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    result = pd.DataFrame()
    result[name] = feature
    feature_noinf = feature[feature < np.inf]
    if lo_pct > 0 or lo_cap is not None:
        lolimit = feature_noinf.quantile(lo_pct)
        if lo_cap is not None:
            lolimit = np.max(lolimit, lo_cap)
        result[name + '_undercap'] = feature < lolimit
        result.loc[feature < lolimit, name] = lolimit
    if up_pct < 1 or up_cap is not None:
        uplimit = feature_noinf.quantile(up_pct)
        if up_cap is not None:
            uplimit = np.min(uplimit, up_cap)
        result[name + '_overcap'] = feature > uplimit
        result.loc[feature > uplimit, name] = uplimit
    return result
