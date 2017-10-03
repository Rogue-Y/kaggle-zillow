# To clean the original features. Will move individual data_clean functions here.

# dummy name generator in notebook "Original data cleaning"

# All the method accept a raw properties datafram (the one load using
# utils.load_properties_data())

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from .utils import *

def parcelid(df):
    return df['parcelid']

def airconditioningtypeid(df):
    # Low non-nan ratio
    # Prior to 1965 there was no AC
    row_indexer = (df['airconditioningtypeid'].isnull()) & (df['yearbuilt'] <= 1965)
    col_indexer = ['airconditioningtypeid']
    df.loc[row_indexer, col_indexer] = 5

    # After 1965 set to "Other"
    # row_indexer = (df['airconditioningtypeid'].isnull()) & (df['yearbuilt'] > 1965)
    # col_indexer = ['airconditioningtypeid']
    # df.loc[row_indexer, col_indexer] = 6
    return df['airconditioningtypeid'].fillna(6)

def architecturalstyletypeid(df):
    # Low non-nan ratio, fill "Other"
    return df['architecturalstyletypeid'].fillna(19)

def basementsqft(df):
    # Low non-nan ratio
    return df['basementsqft'].fillna(0)

def bathroomcnt(df):
    # Median = 2
    return df['bathroomcnt'].fillna(df['bathroomcnt'].median())

def bedroomcnt(df):
    # Median = 3
    return df['bedroomcnt'].fillna(df['bedroomcnt'].median())

def buildingclasstypeid(df):
    # Low non-nan ratio, fill 0 (Not in dict)
    return df['buildingclasstypeid'].fillna(0)

def buildingqualitytypeid(df):
    # 1-12, fill in median or mean = 7
    return df['buildingqualitytypeid'].fillna(df['buildingqualitytypeid'].median())

def calculatedbathnbr(df):
    # 1-20, fill in median or mean = 2
    return df['calculatedbathnbr'].fillna(df['calculatedbathnbr'].median())

# 0.005727, All 66, like a has deck flag
def decktypeid(df):
    # Low non-nan ratio, fill 0 (Not in dict)
    return df['decktypeid'] == 66

def finishedfloor1squarefeet(df):
    # Long tail distribution
    return df['finishedfloor1squarefeet'].fillna(df['finishedfloor1squarefeet'].median())

def calculatedfinishedsquarefeet(df):
    #TODO: see notebook
    return df['calculatedfinishedsquarefeet'].fillna(0)

def finishedsquarefeet12(df):
    # TODO: see notebook
    return df['finishedsquarefeet12'].fillna(0)

def finishedsquarefeet13(df):
    # TODO: see notebook
    return df['finishedsquarefeet13'].fillna(0)

def finishedsquarefeet15(df):
    # TODO: see notebook
    return df['finishedsquarefeet15'].fillna(0)

def finishedsquarefeet50(df):
    # TODO: see notebook
    return df['finishedsquarefeet50'].fillna(0)

def finishedsquarefeet6(df):
    # TODO: see notebook
    return df['finishedsquarefeet6'].fillna(0)

# fireplace
# 0.104728
def fireplacecnt(df):
    # Can fill na with 0
    return df['fireplacecnt'].fillna(0)

# extremly low ratio: 0.001730
# seems has way less data than fireplacecnt
def fireplaceflag(df):
    return df['fireplaceflag'] == True

def has_fireplace(df):
    return df['fireplacecnt'].notnull()


def fullbathcnt(df):
    # Fill in median = 2
    return df['fullbathcnt'].fillna(df['fullbathcnt'].median())


############

def fill_median_and_clip_helper(df, column, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    result = pd.DataFrame()
    feature = df[column].fillna(df[column].median())
    result[column] = feature
    if lo_pct > 0:
        # the percentile is based the values before fill nan
        lolimit = feature.quantile(lo_pct)
        result[column + '_undercap'] = feature < lolimit
        result.loc[result[column] < lolimit, column] = lolimit
    if up_pct < 1:
        uplimit = feature.quantile(up_pct)
        result[column + '_overcap'] = feature > uplimit
        result.loc[result[column] > uplimit, column] = uplimit
    if lo_cap is not None:
        result.loc[result[column] < lo_cap, column] = lo_cap
    if up_cap is not None:
        result.loc[result[column] > up_cap, column] = up_cap
    return result

def fill_value_and_clip_helper(df, column, value=0, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    result = pd.DataFrame()
    feature = df[column].fillna(value)
    result[column] = feature
    if lo_pct > 0:
        # the percentile is based the values before fill nan
        lolimit = feature.quantile(lo_pct)
        result[column + '_undercap'] = feature < lolimit
        result.loc[result[column] < lolimit, column] = lolimit
    if up_pct < 1:
        uplimit = feature.quantile(up_pct)
        result[column + '_overcap'] = feature > uplimit
        result.loc[result[column] > uplimit, column] = uplimit
    if lo_cap is not None:
        result.loc[result[column] < lo_cap, column] = lo_cap
    if up_cap is not None:
        result.loc[result[column] > up_cap, column] = up_cap
    return result


# garage

# up_pct and lo_pct are used to clipping with percentile, should be
# number between 0 and 1
# up_cap and lo_cap are used to clipping with max and min
# should just choose one clipping method
# Non-nan ratio: 0.32966
def garagecarcnt(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'garagecarcnt', lo_pct, up_pct, lo_cap, up_cap)

# Non-nan ratio: 0.32966
# 1500 (~quantile 0.9985) seems a good place to clip
# 182804 rows has garagecnt > 0 but garagesqft = 0 are 0
# could create a is zero feature
def garagetotalsqft(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'garagetotalsqft', lo_pct, up_pct, lo_cap, up_cap)

# def is_garagetotalsqft_zero(df):
#     return df['garagetotalsqft'] == 0
#
# # Some garage has 0 carcnt but non-zero sqft
# def has_partial_garagecarcnt(df):
#     return (df['garagetotalsqft'] > 0) & (df['garagecarcnt'] == 0)


# low non-nan ratio: 0.023119
def hashottuborspa(df):
    return df['hashottuborspa'] == True

# non-nan ratio: 0.907511
def lotsizesquarefeet(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'lotsizesquarefeet', lo_pct, up_pct, lo_cap, up_cap)


# Pool

# 0.173
# all 1, like has pool
def poolcnt(df):
    return df['poolcnt'] == 1

# low ratio: 0.009
def poolsizesum(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'poolsizesum', lo_pct, up_pct, lo_cap, up_cap)

# low ratio: 0.0123
# all 1, means is the pool a spa
def pooltypeid10(df):
    return df['pooltypeid10'] == 1

# low ratio: 0.0107
# all 1, means is the pool with spa
def pooltypeid2(df):
    return df['pooltypeid2'] == 1

# 0.162621
# all 1, means is the pool without hot tub
def pooltypeid7(df):
    return df['pooltypeid7'] == 1

# extremly low ratio: 0.000544
# all 7 basically a is basement flag
def storytypeid(df):
    return df['storytypeid'] == 7

# 0.104
# fill 0 (assume nan means no 3/4 bathrooms)
def threequarterbathnbr(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_value_and_clip_helper(df, 'threequarterbathnbr', 0, lo_pct, up_pct, lo_cap, up_cap)

# 0.662428
# mostly 1
def unitcnt(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'unitcnt', lo_pct, up_pct, lo_cap, up_cap)

# 4 has 39877 count, 5 has only 588
def is_unitcnt_gt_four(df):
    return df['unitcnt'] > 4


# yard
# low ratio: 0.026918
def yardbuildingsqft17(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'yardbuildingsqft17', lo_pct, up_pct, lo_cap, up_cap)

# low ratio: 0.000887
# Storage shed/building in yard
def yardbuildingsqft26(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_value_and_clip_helper(df, 'yardbuildingsqft26', 0, lo_pct, up_pct, lo_cap, up_cap)

# This could be replicate of the above one, since we fill 0 for the above one.
def has_shed_in_yard(df):
    return df['yardbuildingsqft26'].notnull()

# 0.228482
# range from 1 to 41, but only 1 to 4 in the training data
def numberofstories(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'numberofstories', lo_pct, up_pct, lo_cap, up_cap)

def is_numberofstories_gt_three(df):
    return df['numberofstories'] > 3

# tax
# 0.981582
def structuretaxvaluedollarcnt(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'structuretaxvaluedollarcnt', lo_pct, up_pct, lo_cap, up_cap)

# 0.985746
def taxvaluedollarcnt(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'taxvaluedollarcnt', lo_pct, up_pct, lo_cap, up_cap)

# 0.996168
# assume nan value means not assess yet, so fill 2016 (the current year of data)
# TODO: one hot encode this or remove this and only use the boolean feature.
def assessmentyear(df):
    return df['assessmentyear'].fillna(2016)

def is_assessmentyear_2015(df):
    return df['assessmentyear'] == 2015

def is_tax_assessed(df):
    return df['assessmentyear'].notnull()

# 0.977311
def landtaxvaluedollarcnt(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'landtaxvaluedollarcnt', lo_pct, up_pct, lo_cap, up_cap)

# 0.989532
def taxamount(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'taxamount', lo_pct, up_pct, lo_cap, up_cap)

# 0.018914
# boolean
def taxdelinquencyflag(df):
    return df['taxdelinquencyflag'] == 'Y'

# 0.018915
# nan means tax is paid, so put in 2016 to represent that.
# Also change the value to YYYY format
def taxdelinquencyyear(df):
    tmp = df['taxdelinquencyyear'].fillna(16)
    result = tmp + 1900
    # get the right year for 2000 to 2017
    return result.where(tmp > 17, result + 100)

# Also try buckets for this one.
def is_taxdelinquencyyear_before_2014(df):
    return taxdelinquencyyear(df) < 2014

# The difference between the sum of structure and land tax with total tax
def tax_difference_with_nan(df):
    return (df['structuretaxvaluedollarcnt'] + df['landtaxvaluedollarcnt']
        - df['taxvaluedollarcnt'])

def tax_difference_fill_nan(df):
    return (structuretaxvaluedollarcnt(df) + landtaxvaluedollarcnt(df)
        - taxvaluedollarcnt(df))

# Categorical:
# non-nan ratio: 0.605115
def heatingorsystemtypeid(df):
    # Consider this is LA, fill with None(code 13)
    return df['heatingorsystemtypeid'].fillna(13)

# low ratio: 0.002260
def typeconstructiontypeid(df):
    # fill 0 to mean unknown
    return df['typeconstructiontypeid'].fillna(0)

def has_construction_type(df):
    return df['typeconstructiontypeid'].notnull()

# Geo related categorical
# 0.99410
def propertycountylandusecode(df):
    # fill with mode for now, look into if we could use geo to fill it
    filled = df['propertycountylandusecode'].fillna(
        df['propertycountylandusecode'].mode()[0])
    labelEncoder = LabelEncoder()
    return pd.Series(labelEncoder.fit_transform(filled.astype(str)))

# 0.64214
def propertylandusetypeid(df):
    return df['propertylandusetypeid'].fillna(df['propertylandusetypeid'].mode()[0])

def propertyzoningdesc(df):
    # Use to mode in the land use type id group to fill most, than fill the rest
    # with global mode
    # TODO: try a easier way to fill and compare
    mode = df['propertyzoningdesc'].mode()[0]
    id_desc = df['propertyzoningdesc'].groupby(df['propertylandusetypeid']).value_counts()
    group_mode_df = id_desc.groupby(level=0).nlargest(1)
    group_mode_dict = dict(group_mode_df.reset_index(level=1).index)
    # id 265, 270, 275's corresponding desc are all nan, so create fake desc for
    # those parcels
    group_mode_dict[265.0] = '265'
    group_mode_dict[270.0] = '270'
    group_mode_dict[275.0] = '275'
    id_desc_map = df['propertylandusetypeid'].map(group_mode_dict)

    desc_list = df['propertyzoningdesc'].where(
        df['propertyzoningdesc'].notnull(), id_desc_map).fillna(mode)
    labelEncoder = LabelEncoder()
    return pd.Series(labelEncoder.fit_transform(desc_list.astype(str)))



# Geo
# TODO: besides lat and lon, other geo features need to be one-hot encoded
# nan latitude and longitude should be removed when loading properties dataset
def latitude(df):
    return df['latitude']

def longitude(df):
    return df['longitude']

# use 0 to mean nan geo features
def regionidcity(df):
    return df['regionidcity'].fillna(0)

def regionidcounty(df):
    return df['regionidcounty'].fillna(0)

def regionidneighborhood(df):
    return df['regionidneighborhood'].fillna(0)

def regionidzip(df):
    return df['regionidzip'].fillna(0)

def fips(df):
    # fill in mode "6037"
    return df['fips'].fillna(df['fips'].mode()[0])

# TODO:
#   OBSOLETED

#   1. parse censustractandblock features
#   2. No dominant modes, could use geo to help fill, fill 0 for now.
# 0.99410
def rawcensustractandblock(df):
    return df['rawcensustractandblock'].fillna(0)

# 0.98744
def censustractandblock(df):
    return df['censustractandblock'].fillna(0)



# Other features that need discussion

# Really wired
# 0.996
# mostly 0 and median is 0,
# could replace this with the sum of bedroom and bathroom
def roomcnt(df, lo_pct=0, up_pct=1, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'roomcnt', lo_pct, up_pct, lo_cap, up_cap)

def is_roomcnt_zero(df):
    return df['roomcnt'] == 0

def total_room_fill_nan(df):
    return bedroomcnt(df) + bathroomcnt(df)

def total_room_with_nan(df):
    return df['bathroomcnt'] + df['bedroomcnt']


# 0.979925
# TODO: There's no dominant mode in data. Maybe we could 1. infer the year from other
# features; 2. for each parcel, randomly choose a year from the top n counts.
def yearbuilt(df):
    # For know, fill 0 to mean unknown.
    return df['yearbuilt'].fillna(0)


def propertycountylandusecode_cat(df):
    return df['propertycountylandusecode_cat'].fillna(0)

def fips_census_1(df):
    return df['fips_census_1'].fillna(df['fips_census_1'].mode()[0])

def block_1(df):
    return df['block_1'].fillna(0)

def fips_census_block(df):
    return df['fips_census_block'].fillna(0)
