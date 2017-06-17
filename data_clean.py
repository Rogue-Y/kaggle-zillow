import pandas as pd
import numpy as np

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
