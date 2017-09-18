# To clean the original features. Will move individual data_clean functions here.

# dummy name generator in notebook "Original data cleaning"

import pandas as pd
from .utils import *
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder

def parcelid(df):
    return df['parcelid']

def airconditioningtypeid(df):
    return df['airconditioningtypeid']

def architecturalstyletypeid(df):
    return df['architecturalstyletypeid']

def basementsqft(df):
    return df['basementsqft']

def bathroomcnt(df):
    return df['bathroomcnt']

def bedroomcnt(df):
    return df['bedroomcnt']

def buildingclasstypeid(df):
    return df['buildingclasstypeid']

def buildingqualitytypeid(df):
    return df['buildingqualitytypeid']

def calculatedbathnbr(df):
    return df['calculatedbathnbr']

def decktypeid(df):
    return df['decktypeid']

def finishedfloor1squarefeet(df):
    return df['finishedfloor1squarefeet']

def calculatedfinishedsquarefeet(df):
    return df['calculatedfinishedsquarefeet']

def finishedsquarefeet12(df):
    return df['finishedsquarefeet12']

def finishedsquarefeet13(df):
    return df['finishedsquarefeet13']

def finishedsquarefeet15(df):
    return df['finishedsquarefeet15']

def finishedsquarefeet50(df):
    return df['finishedsquarefeet50']

def finishedsquarefeet6(df):
    return df['finishedsquarefeet6']

def fips(df):
    return df['fips']

def fireplacecnt(df):
    return df['fireplacecnt']

def fullbathcnt(df):
    return df['fullbathcnt']

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