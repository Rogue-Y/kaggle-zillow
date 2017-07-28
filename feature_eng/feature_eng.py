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

def get_distance(p1, p2):
    """ This use a rough calculation of distance, will overestimate the distance
        for latitude around 30 degree
        for more, see: http://www.movable-type.co.uk/scripts/latlong.html
        the lat / lon in data set divided by 10^6 is degree, and need to be
        converted to radian for distance calculation
        the calculated distance is in km
    """
    # Radius of the earth
    R = 6371;
    pi = 3.14
    scale = 10**6
    diff_lat = (p1[0] - p2[0]) * pi / scale / 180
    diff_lon = (p1[1] - p2[1]) * pi / scale / 180
    return math.hypot(diff_lat, diff_lon) * R

def add_nearby_houses(df, threshold=5):
    """ Calulate how many nearby houses within threshold km
        Returns:
            a copy of df with new feature added.
    """
    rows = df.shape[0]
    df['nearby_houses'] = np.zeros(rows)
    df['nearby_houses'] = df['nearby_houses'].astype(int)
    for i in range(rows):
        nearby_houses_i = df.loc[i, 'nearby_houses']
        for j in range(i, rows):
            p1 = (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
            p2 = (df.loc[j, 'latitude'], df.loc[j, 'longitude'])
            if get_distance(p1, p2) < threshold:
                nearby_houses_i = nearby_houses_i + 1
                df.loc[j, 'nearby_houses'] = df.loc[j, 'nearby_houses'] + 1
        df.loc[i, 'nearby_houses'] = nearby_houses_i
    df.to_csv('data/add_nearby_houses.csv', index=True)
    return df

def add_bins(df, year_bin_num=10, tax_bin_num=10):
    """ Put some features into bins.
        Returns:
            a copy of df with bins columns added.
        Columns:
            ['yearbuilt', 'taxamount']
    """

    # TODO(hzn): think about if use qcut on the whole data set will cause data
    # leakage
    # yearbuilt
    year_bins = pd.qcut(df['yearbuilt'], year_bin_num, labels=False)
    year_dummies = pd.get_dummies(year_bins, prefix="yearbuilt")

    # taxamount
    tax_bins = pd.qcut(df['taxamount'], tax_bin_num, labels=False)
    tax_dummies = pd.get_dummies(tax_bins, prefix="taxamount")

    df_list = [df, year_dummies, tax_dummies]
    return pd.concat(df_list, axis=1)
