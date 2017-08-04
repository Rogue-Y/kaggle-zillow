# feature engineering
import pandas as pd
import numpy as np
import math

def add_missing_value_count(df):
    df['missing_values'] = df.isnull().sum(axis=1)
    return df

def convert_year_build_to_age(df):
    df['yearbuilt'] = 2017 - df['yearbuilt']
    return df

def add_before_1900_column(df):
    df['is_before_1900'] = df['yearbuilt'] <= 1900
    return df

# TODO(hzn): investigate data leakage in these new features
def add_features(df):
    # From https://www.kaggle.com/nikunjm88/creating-additional-features
    #life of property
    df['N-life'] = 2018 - df['yearbuilt']

    #error in calculation of the finished living area of home
    df['N-LivingAreaError'] = df['calculatedfinishedsquarefeet']/df['finishedsquarefeet12']

    #proportion of living area
    df['N-LivingAreaProp'] = df['calculatedfinishedsquarefeet']/df['lotsizesquarefeet']

    #Amout of extra space
    df['N-ExtraSpace'] = df['lotsizesquarefeet'] - df['calculatedfinishedsquarefeet']

    #Total number of rooms
    df['N-TotalRooms'] = df['bathroomcnt']*df['bedroomcnt']

    #Average room size
    df['N-AvRoomSize'] = df['calculatedfinishedsquarefeet']/df['roomcnt']

    # Number of Extra rooms
    df['N-ExtraRooms'] = df['roomcnt'] - df['N-TotalRooms']

    #Ratio of the built structure value to land area
    df['N-ValueProp'] = df['structuretaxvaluedollarcnt']/df['landtaxvaluedollarcnt']

    #Does property have a garage, pool or hot tub and AC?
    df['N-GarPoolAC'] = ((df['garagecarcnt']>0) & (df['pooltypeid10']>0) & (df['airconditioningtypeid']!=5))*1

    df["N-location"] = df["latitude"] + df["longitude"]
    df["N-location-2"] = df["latitude"]*df["longitude"]
    df["N-location-2round"] = df["N-location-2"].round(-4)

    df["N-latitude-round"] = df["latitude"].round(-4)
    df["N-longitude-round"] = df["longitude"].round(-4)

    # Tax related
    #Ratio of tax of property over parcel
    df['N-ValueRatio'] = df['taxvaluedollarcnt']/df['taxamount']

    #TotalTaxScore
    df['N-TaxScore'] = df['taxvaluedollarcnt']*df['taxamount']

    # Geo
    #Number of properties in the zip
    zip_count = df['regionidzip'].value_counts().to_dict()
    df['N-zip_count'] = df['regionidzip'].map(zip_count)

    #Number of properties in the city
    city_count = df['regionidcity'].value_counts().to_dict()
    df['N-city_count'] = df['regionidcity'].map(city_count)

    #Number of properties in the county
    region_count = df['regionidcounty'].value_counts().to_dict()
    df['N-county_count'] = df['regionidcounty'].map(city_count)

    # others
    #Indicator whether it has AC or not
    df['N-ACInd'] = (df['airconditioningtypeid']!=5)*1

    #Indicator whether it has Heating or not
    df['N-HeatInd'] = (df['heatingorsystemtypeid']!=13)*1

    #There's 25 different property uses - let's compress them down to 4 categories
    #df['N-PropType'] = df.propertylandusetypeid.replace({31 : "Mixed", 46 : "Other", 47 : "Mixed", 246 : "Mixed", 247 : "Mixed", 248 : "Mixed", 260 : "Home", 261 : "Home", 262 : "Home", 263 : "Home", 264 : "Home", 265 : "Home", 266 : "Home", 267 : "Home", 268 : "Home", 269 : "Not Built", 270 : "Home", 271 : "Home", 273 : "Home", 274 : "Other", 275 : "Home", 276 : "Home", 279 : "Home", 290 : "Not Built", 291 : "Not Built" })
    #polnomials of the variable
    df["N-structuretaxvaluedollarcnt-2"] = df["structuretaxvaluedollarcnt"] ** 2
    df["N-structuretaxvaluedollarcnt-3"] = df["structuretaxvaluedollarcnt"] ** 3

    #Average structuretaxvaluedollarcnt by city
    group = df.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
    df['N-Avg-structuretaxvaluedollarcnt'] = df['regionidcity'].map(group)

    #Deviation away from average
    df['N-Dev-structuretaxvaluedollarcnt'] = abs((df['structuretaxvaluedollarcnt'] - df['N-Avg-structuretaxvaluedollarcnt']))/df['N-Avg-structuretaxvaluedollarcnt']

    return df

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
            df (not necessarily a copy) with bins columns(optionally one-hot encoded) added.
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
    return pd.concat([df, bin_dummies], axis=1)

def add_year_tax_bins(df, year_bin_num=10, tax_bin_num=10):
    """ Put some features into bins.
        Returns:
            a copy of df with bins columns added.
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
            the dataframe (not a copy) with crossed columns added.
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
            a copy of dataframe with crossed columns added.
    """
    df = df.copy()
    for feature_tuple in cross_list:
        cross_features(df, *feature_tuple)

    return df
