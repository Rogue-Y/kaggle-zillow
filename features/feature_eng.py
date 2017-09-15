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
    return df['calculatedfinishedsquarefeet']/df['finishedsquarefeet12']

def error_rate_first_floor_living_sqft(df):
    return df['finishedsquarefeet50']/df['finishedfloor1squarefeet']

def error_rate_bathroom(df):
    return df['calculatedbathnbr']/df['bathroomcnt']

def error_rate_count_bathroom(df):
    return (df['threequarterbathnbr'] + df['fullbathcnt'])/df['bathroomcnt']

def ratio_living_area(df):
    #proportion of living area
    return df['calculatedfinishedsquarefeet']/df['lotsizesquarefeet']

def ratio_living_area_2(df):
    return df['finishedsquarefeet12']/df['finishedsquarefeet15']

def ratio_bedroom_bathroom(df):
    return df['bedroomcnt']/df['bathroomcnt']

def ratio_basement(df):
    return df['basementsqft']/df['finishedsquarefeet12']

def ratio_pool_yard(df):
    return df['poolsizesum']/df['yardbuildingsqft17']

def ratio_pool_shed(df):
    return df['yardbuildingsqft26']/df['yardbuildingsqft17']

def ratio_floor_shape(df):
    return df['finishedsquarefeet13']/df['calculatedfinishedsquarefeet']

def ratio_fireplace(df):
    return df['fireplacecnt']/df['finishedsquarefeet15']

def extra_space(df):
    #Amout of extra space
    return df['lotsizesquarefeet'] - df['calculatedfinishedsquarefeet']

def total_rooms(df):
    #Total number of rooms
    return df['bathroomcnt'] + df['bedroomcnt']

def average_room_size(df):
    #Average room size
    total = total_rooms(df)
    return df['calculatedfinishedsquarefeet']/total

# # roomcnt is mostly 0
# def average_room_size_2(df):
#     return df['calculatedfinishedsquarefeet']/df['roomcnt']

def average_bathroom_size(df):
    return df['finishedsquarefeet12']/df['bathroomcnt']

def average_bedroom_size(df):
    return df['finishedsquarefeet12']/df['bedroomcnt']

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
    return df['taxvaluedollarcnt']/df['landtaxvaluedollarcnt']

def ratio_structure_tax_value_to_land_tax_value(df):
    #Ratio of the built structure value to land area
    return df['structuretaxvaluedollarcnt']/df['landtaxvaluedollarcnt']

def ratio_tax(df):
    #Ratio of tax of property over parcel
    return df['taxamount']/df['taxvaluedollarcnt']

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
        neighborhood[col + '_neighborhood_mean_ratio'] = df[col] / neighborhood[col + '_neighborhood_mean']
        neighborhood[col + '_neighborhood_std'] = df['regionidneighborhood'].map(neighborhood_dict[(col, 'std')])
        neighborhood[col + '_neighborhood_std_ratio'] = (df[col] - neighborhood[col + '_neighborhood_mean']) / neighborhood[col + '_neighborhood_std']
        # neighborhood.drop(col + '_neighborhood_mean', axis=1, inplace=True)
        # neighborhood.drop(col + '_neighborhood_std', axis=1, inplace=True)
        # neighborhood[col + '_neighborhood_max'] = df['regionidneighborhood'].map(neighborhood_dict[(col, 'max')])
        # neighborhood[col + '_neighborhood_min'] = df['regionidneighborhood'].map(neighborhood_dict[(col, 'min')])

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
        city[col + '_city_std'] = df['regionidcity'].map(city_dict[(col, 'std')])
        city[col + '_city_mean_ratio'] = df[col] / city[col + '_city_mean']
        city[col + '_city_std_ratio'] = (df[col] - city[col + '_city_mean']) / city[col + '_city_std']
        # city[col + '_city_max'] = df['regionidcity'].map(city_dict[(col, 'max')])
        # city[col + '_city_min'] = df['regionidcity'].map(city_dict[(col, 'min')])

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
        zip[col + '_zip_std'] = df['regionidzip'].map(zip_dict[(col, 'std')])
        zip[col + '_zip_mean_ratio'] = df[col] / zip[col + '_zip_mean']
        zip[col + '_zip_std_ratio'] = (df[col] - zip[col + '_zip_mean']) / zip[col + '_zip_std']
        # zip[col + '_zip_max'] = df['regionidzip'].map(zip_dict[(col, 'max')])
        # zip[col + '_zip_min'] = df['regionidzip'].map(zip_dict[(col, 'min')])

    return zip

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
        county[col + '_county_std'] = df['regionidcounty'].map(county_dict[(col, 'std')])
        county[col + '_county_mean_ratio'] = df[col] / county[col + '_county_mean']
        county[col + '_county_std_ratio'] = (df[col] - county[col + '_county_mean']) / county[col + '_county_std']
        # county[col + '_county_max'] = df['regionidcounty'].map(county_dict[(col, 'max')])
        # county[col + '_county_min'] = df['regionidcounty'].map(county_dict[(col, 'min')])

    return county

# def geo_neighborhood_tax_value(df):
#     neighborhood_df = pd.DataFrame()
#     #Number of properties in the neighborhood
#     neighborhood_count = df['regionidneighborhood'].value_counts().to_dict()
#     neighborhood_df['neighborhood_count'] = df['regionidneighborhood'].map(neighborhood_count)

#     # stats of value estimate of properties grouped by neighborhood
#     neighborhood_dict = (df[['regionidneighborhood', 'taxvaluedollarcnt']].groupby('regionidneighborhood')
#         .agg(['max', 'min', 'std', 'mean'])['taxvaluedollarcnt'].to_dict())

#     neighborhood_df['neighborhood_value_mean'] = df['regionidneighborhood'].map(neighborhood_dict['mean'])
#     neighborhood_df['neighborhood_value_std'] = df['regionidneighborhood'].map(neighborhood_dict['std'])
#     neighborhood_df['neighborhood_value_max'] = df['regionidneighborhood'].map(neighborhood_dict['max'])
#     neighborhood_df['neighborhood_value_min'] = df['regionidneighborhood'].map(neighborhood_dict['min'])
#     neighborhood_df['neighborhood_value_range'] = neighborhood_df['neighborhood_value_max'] - neighborhood_df['neighborhood_value_min']

#     return neighborhood_df

# def geo_neighborhood_tax_value_ratio_mean(df):
#     neighborhood_df = geo_neighborhood_tax_value(df)
#     return df['taxvaluedollarcnt'] / neighborhood_df['neighborhood_value_mean']

# def geo_zip_tax_value(df):
#     zip_df = pd.DataFrame()
#     #Number of properties in the zip
#     zip_count = df['regionidzip'].value_counts().to_dict()
#     zip_df['zip_count'] = df['regionidzip'].map(zip_count)

#     # stats of value estimate of properties grouped by zip
#     zip_dict = (df[['regionidzip', 'taxvaluedollarcnt']].groupby('regionidzip')
#         .agg(['max', 'min', 'std', 'mean'])['taxvaluedollarcnt'].to_dict())

#     zip_df['zip_value_mean'] = df['regionidzip'].map(zip_dict['mean'])
#     zip_df['zip_value_std'] = df['regionidzip'].map(zip_dict['std'])
#     zip_df['zip_value_max'] = df['regionidzip'].map(zip_dict['max'])
#     zip_df['zip_value_min'] = df['regionidzip'].map(zip_dict['min'])
#     zip_df['zip_value_range'] = zip_df['zip_value_max'] - zip_df['zip_value_min']

#     return zip_df

# def geo_city_tax_value(df):
#     city = pd.DataFrame()
#     #Number of properties in the city
#     city_count = df['regionidcity'].value_counts().to_dict()
#     city['city_count'] = df['regionidcity'].map(city_count)

#     # stats of value estimate of properties grouped by city
#     city_dict = (df[['regionidcity', 'taxvaluedollarcnt']].groupby('regionidcity')
#         .agg(['max', 'min', 'std', 'mean'])['taxvaluedollarcnt'].to_dict())

#     city['city_value_mean'] = df['regionidcity'].map(city_dict['mean'])
#     city['city_value_std'] = df['regionidcity'].map(city_dict['std'])
#     city['city_value_max'] = df['regionidcity'].map(city_dict['max'])
#     city['city_value_min'] = df['regionidcity'].map(city_dict['min'])
#     city['city_value_range'] = city['city_value_max'] - city['city_value_min']

#     return city

# def geo_region_tax_value(df):
#     region = pd.DataFrame()
#     #Number of properties in the county
#     region_count = df['regionidcounty'].value_counts().to_dict()
#     region['county_count'] = df['regionidcounty'].map(region_count)

#     # stats of value estimate of properties grouped by county
#     county_dict = (df[['regionidcounty', 'taxvaluedollarcnt']].groupby('regionidcounty')
#         .agg(['max', 'min', 'std', 'mean'])['taxvaluedollarcnt'].to_dict())

#     region['county_value_mean'] = df['regionidcounty'].map(county_dict['mean'])
#     region['county_value_std'] = df['regionidcounty'].map(county_dict['std'])
#     region['county_value_max'] = df['regionidcounty'].map(county_dict['max'])
#     region['county_value_min'] = df['regionidcounty'].map(county_dict['min'])
#     region['county_value_range'] = region['county_value_max'] - region['county_value_min']

#     return region

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

    values = df[columns]
    values['lat_lon_block'] = blocks

    # stats of value estimate of properties grouped by lat_lon_block
    lat_lon_block_dict = (values.groupby('lat_lon_block')
        .agg(['max', 'min', 'std', 'mean']).to_dict())

    for col in columns:
        lat_lon_block[col + '_lat_lon_block_mean'] = blocks.map(lat_lon_block_dict[(col, 'mean')])
        lat_lon_block[col + '_lat_lon_block_mean_ratio'] = df[col] / lat_lon_block[col + '_lat_lon_block_mean']
        lat_lon_block[col + '_lat_lon_block_std'] = blocks.map(lat_lon_block_dict[(col, 'std')])
        lat_lon_block[col + '_lat_lon_block_std_ratio'] = (df[col] - lat_lon_block[col + '_lat_lon_block_mean']) / lat_lon_block[col + '_lat_lon_block_std']
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

def category_ac_type_encode(df):
    return pd.Series(labelEncoder.fit_transform(df['airconditioningtypeid']))

def category_ac_type_one_hot(df):
    return pd.get_dummies(df['airconditioningtypeid'])

def category_fips_type_encode(df):
    return pd.Series(labelEncoder.fit_transform(df['fips']))

def category_fips_type_one_hot(df):
    return pd.get_dummies(df['fips'])

def category_heating_type_encode(df):
    return pd.Series(labelEncoder.fit_transform(df['heatingorsystemtypeid']))

def category_heating_type_one_hot(df):
    return pd.get_dummies(df['heatingorsystemtypeid'])

def category_land_use_code_helper(df):
    land_use_code_threshold = 50000
    luc_vc = df['propertycountylandusecode'].value_counts()
    land_use_code = df['propertycountylandusecode'].mask(df['propertycountylandusecode'].map(luc_vc) < land_use_code_threshold, 'others')
    return land_use_code.astype('str')

def category_land_use_code_encode(df):
    return pd.Series(labelEncoder.fit_transform(category_land_use_code_helper(df)))

def category_land_use_code_one_hot(df):
    return pd.get_dummies(category_land_use_code_helper(df))

def category_land_use_desc_helper(df):
    land_use_desc_threshold = 10000
    lud_vc = df['propertyzoningdesc'].value_counts()
    land_use_desc = df['propertyzoningdesc'].mask(df['propertyzoningdesc'].map(lud_vc) < land_use_desc_threshold, 'others')
    return land_use_desc.astype('str')

def category_land_use_desc_encode(df):
    return pd.Series(labelEncoder.fit_transform(category_land_use_desc_helper(df)))

def category_land_use_desc_one_hot(df):
    return pd.get_dummies(category_land_use_desc_helper(df))

#polnomials of the variable
def poly_2_structure_tax_value(df):
    return df["structuretaxvaluedollarcnt"] ** 2

def poly_3_structure_tax_value(df):
    return df["structuretaxvaluedollarcnt"] ** 3

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
