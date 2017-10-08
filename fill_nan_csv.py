import pandas as pd

infered_type = {
    "rawcensustractandblock": str,
    "censustractandblock": str,
    "propertycountylandusecode": str
}

prop2016 = pd.read_csv('data/properties_2016.csv', dtype=infered_type)
prop2017 = pd.read_csv('data/properties_2017.csv', dtype=infered_type)

nan_row_2016 = prop2016[prop2016['latitude'].isnull() | prop2016['longitude'].isnull()][['parcelid']]
nan_row_2017 = prop2017[prop2017['latitude'].isnull() | prop2017['longitude'].isnull()][['parcelid']]

fill_row_2016 = nan_row_2016.merge(prop2017, 'left', 'parcelid')
fill_row_2016.set_index(nan_row_2016.index, inplace=True)

fill_row_2017 = nan_row_2017.merge(prop2016, 'left', 'parcelid')
fill_row_2017.set_index(nan_row_2017.index, inplace=True)

prop2016 = prop2016[prop2016['latitude'].notnull() & prop2016['longitude'].notnull()]
prop2017 = prop2017[prop2017['latitude'].notnull() & prop2017['longitude'].notnull()]

prop2016 = pd.concat([prop2016, fill_row_2016]).sort_index()
prop2017 = pd.concat([prop2017, fill_row_2017]).sort_index()

prop2017.to_csv('data/properties_2017_filled.csv', index=False)
prop2016.to_csv('data/properties_2016_filled.csv', index=False)
prop = pd.read_csv('data/properties_2016_filled.csv', dtype=infered_type)
