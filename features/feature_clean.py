# To clean the original features. Will move individual data_clean functions here.

# dummy name generator in notebook "Original data cleaning"

import pandas as pd
from .utils import *
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder


def airconditioningtypeid(df):
    # Low non-nan ratio
    # Prior to 1965 there was no AC
    row_indexer = (df['airconditioningtypeid'].isnull()) & (df['yearbuilt'] <= 1965)
    col_indexer = ['airconditioningtypeid']
    df.loc[row_indexer, col_indexer] = 5

    # After 1965 set to "Other"
    row_indexer = (df['airconditioningtypeid'].isnull()) & (df['yearbuilt'] > 1965)
    col_indexer = ['airconditioningtypeid']
    df.loc[row_indexer, col_indexer] = 6
    return df['airconditioningtypeid']

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
    return df['calculatedfinishedsquarefeet'].fillna()

def finishedsquarefeet12(df):
    return df['finishedsquarefeet12'].fillna()

def finishedsquarefeet13(df):
    return df['finishedsquarefeet13'].fillna()

def finishedsquarefeet15(df):
    return df['finishedsquarefeet15'].fillna()

def finishedsquarefeet50(df):
    return df['finishedsquarefeet50'].fillna()

def finishedsquarefeet6(df):
    return df['finishedsquarefeet6'].fillna()

def fips(df):
    # fill in mode "6037"
    return df['fips'].fillna(df['fips'].mode()[0])

def fireplacecnt(df):
    # Can fill na with 0
    return df['fireplacecnt'].fillna(0)

def fullbathcnt(df):
    # Fill in median = 2
    return df['fullbathcnt'].fillna(df['fullbathcnt'].median())

def garagecarcnt(df):
    return df['garagecarcnt']

def garagetotalsqft(df):
    return df['garagetotalsqft']

def hashottuborspa(df):
    return df['hashottuborspa']

def heatingorsystemtypeid(df):
    return df['heatingorsystemtypeid']

def latitude(df):
    return df['latitude']

def longitude(df):
    return df['longitude']

def lotsizesquarefeet(df):
    return df['lotsizesquarefeet']

def poolcnt(df):
    return df['poolcnt']

def poolsizesum(df):
    return df['poolsizesum']

def pooltypeid10(df):
    return df['pooltypeid10']

def pooltypeid2(df):
    return df['pooltypeid2']

def pooltypeid7(df):
    return df['pooltypeid7']

def propertycountylandusecode(df):
    return df['propertycountylandusecode']

def propertylandusetypeid(df):
    return df['propertylandusetypeid']

def propertyzoningdesc(df):
    return df['propertyzoningdesc']

def rawcensustractandblock(df):
    return df['rawcensustractandblock']

def regionidcity(df):
    return df['regionidcity']

def regionidcounty(df):
    return df['regionidcounty']

def regionidneighborhood(df):
    return df['regionidneighborhood']

def regionidzip(df):
    return df['regionidzip']

def roomcnt(df):
    return df['roomcnt']

def storytypeid(df):
    return df['storytypeid']

def threequarterbathnbr(df):
    return df['threequarterbathnbr']

def typeconstructiontypeid(df):
    return df['typeconstructiontypeid']

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