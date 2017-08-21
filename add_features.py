# Add features one by one and see if they work
import gc

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
import xgboost as xgb

from features import utils
from features import feature_combine
from features import data_clean
from evaluator import Evaluator

# folds number of K-Fold
FOLDS = 5

def train(train, prop, feature_list):
    # feature engineering
    print('Feature engineering')
    prop = feature_combine.feature_combine(
        prop, feature_list, False, 'features/feature_pickles/')
    print(prop.shape)

    # merge transaction and prop data
    df = train.merge(prop, how='left', on='parcelid')

    # split by date
    train_q1_q3, train_q4 = utils.split_by_date(df)
    del df; gc.collect()

    train_q1_q3 = data_clean.drop_training_only_column(train_q1_q3)
    train_q4 = data_clean.drop_training_only_column(train_q4)
    X_train_q1_q3, y_train_q1_q3 = utils.get_features_target(train_q1_q3)
    X_train_q4, y_train_q4 = utils.get_features_target(train_q4)
    del train_q1_q3; del train_q4; gc.collect()

    # model
    xgb_params = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': 0.4,
        # 'base_score': y_mean,
        'silent': 1
    }
    num_boost_rounds = 250

    # split train_q4 into k folds, each time combine k-1 folds with train_q1_q3
    # to train model and validate on the left out fold
    print('training...')
    mean_errors = []
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    # X_train_q4.to_csv('test_train_q4.csv')
    for train_index, validate_index in kf.split(X_train_q4):
        X_train = pd.concat([X_train_q1_q3, X_train_q4.iloc[train_index]], ignore_index=True)
        y_train = pd.concat([y_train_q1_q3, y_train_q4.iloc[train_index]], ignore_index=True)
        # TODO(hzn): add training preprocessing, like remove outliers, resampling

        X_validate = X_train_q4.iloc[validate_index]
        y_validate = y_train_q4.iloc[validate_index]

        d_train = xgb.DMatrix(X_train, y_train)
        model = xgb.train(dict(xgb_params, silent=1), d_train,
            num_boost_round=num_boost_rounds)

        d_validate = xgb.DMatrix(X_validate)
        y_pred = model.predict(d_validate)
        # TODO(hzn): add output training records
        mae = Evaluator.mean_error(y_pred, y_validate)
        mean_errors.append(mae)

    average_error = np.mean(mean_errors)
    print("average cross validation mean error", average_error)
    return average_error

if __name__ == '__main__':
    # Feature list
    from features import test_feature_list
    feature_list = test_feature_list.feature_list

    # load training data
    print('Load training data...')
    train_df, prop = utils.load_train_data()
    prop = data_clean.clean_boolean_data(prop)
    prop = data_clean.cat2num(prop)
    current_mae = train(train_df, prop, [])
    useful_features = []
    for feature in feature_list:
        print('Testing ' + feature[0])
        avg_mae = train(train_df, prop, useful_features+[feature])
        if avg_mae < current_mae:
            current_mae = avg_mae
            useful_features.append(feature)

    # print useful feature names:
    for feature in useful_features:
        print(feature[0])
