# feature engineering
import pandas as pd
import numpy as np

def add_missing_value_count(df):
    df['missing_values'] = df.isnull().sum(axis=1)
    return df
