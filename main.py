import data_clean
import feature_eng
import utils
import base
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

print('Loading data ...')
train, prop, df = utils.load_train_data()
df = feature_eng.add_missing_value_count(df)
df = data_clean.clean_categorical_data(df)
df = data_clean.clean_boolean_data(df)
df = data_clean.drop_columns(df, isTrain=True)

X, y = utils.get_features_target(df)
X_train, X_valid, y_train, y_valid = utils.train_valid_split(X, y, 0.2)
predictor = base.train_lgb(X_train, y_train, X_valid, y_valid);
print('Predicting ...')
test_df = feature_eng.add_missing_value_count(test_df)
test_df, sample = utils.load_test_data(prop)
test_df = data_clean.clean_categorical_data(test_df)
test_df = data_clean.clean_boolean_data(test_df)
X_test = data_clean.drop_columns(test_df, isTrain=False)
utils.predict(predictor, X_test, sample, suffix='0713')
