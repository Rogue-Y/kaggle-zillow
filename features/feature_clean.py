# To clean the original features. Will move individual data_clean functions here.

# dummy name generator in notebook "Original data cleaning"

import pandas as pd
import numpy as np
import math


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

def decktypeid(df):
    # Low non-nan ratio, fill 0 (Not in dict)
    return df['decktypeid'].fillna(0)

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

def fips(df):
    # fill in mode "6037"
    return df['fips'].fillna(df['fips'].mode()[0])

def fireplacecnt(df):
    # Can fill na with 0
    return df['fireplacecnt'].fillna(0)

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
def garagetotalsqft(df, lo_pct=0, up_pct=0.99, lo_cap=None, up_cap=None):
    return fill_median_and_clip_helper(df, 'garagetotalsqft', lo_pct, up_pct, lo_cap, up_cap)

def is_garagetotalsqft_zero(df):
    return df['garagetotalsqft'] == 0

# Some garage has 0 carcnt but non-zero sqft
def has_partial_garagecarcnt(df):
    return (df['garagetotalsqft'] > 0) & (df['garagecarcnt'] == 0)


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
    return fill_value_and_clip_helper(df, threequarterbathnbr, 0, lo_pct, up_pct, lo_cap, up_cap)

def unitcnt(df):
    return df['unitcnt']

def yardbuildingsqft17(df):
    return df['yardbuildingsqft17']

def yardbuildingsqft26(df):
    return df['yardbuildingsqft26']

def yearbuilt(df):
    return df['yearbuilt']

def numberofstories(df):
    return df['numberofstories']

def fireplaceflag(df):
    return df['fireplaceflag']

def structuretaxvaluedollarcnt(df):
    return df['structuretaxvaluedollarcnt']

def taxvaluedollarcnt(df):
    return df['taxvaluedollarcnt']

def assessmentyear(df):
    return df['assessmentyear']

def landtaxvaluedollarcnt(df):
    return df['landtaxvaluedollarcnt']

def taxamount(df):
    return df['taxamount']

def taxdelinquencyflag(df):
    return df['taxdelinquencyflag']

def taxdelinquencyyear(df):
    return df['taxdelinquencyyear']

def censustractandblock(df):
    return df['censustractandblock']


# Categorical:
# non-nan ratio: 0.605115
def heatingorsystemtypeid(df):
    return df['heatingorsystemtypeid']

def propertycountylandusecode(df):
    return df['propertycountylandusecode']

def propertylandusetypeid(df):
    return df['propertylandusetypeid']

def propertyzoningdesc(df):
    return df['propertyzoningdesc']

def rawcensustractandblock(df):
    return df['rawcensustractandblock']

# low ratio: 0.002260
def typeconstructiontypeid(df):
    return df['typeconstructiontypeid']

# Geo
def latitude(df):
    return df['latitude']

def longitude(df):
    return df['longitude']

def regionidcity(df):
    return df['regionidcity']

def regionidcounty(df):
    return df['regionidcounty']

def regionidneighborhood(df):
    return df['regionidneighborhood']

def regionidzip(df):
    return df['regionidzip']


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