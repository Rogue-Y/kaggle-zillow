import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Configuration:
# folds number of K-Fold
FOLDS = 5

# Feature list
from features import test_feature_list
feature_list = test_feature_list.feature_list

# model
from models import XGBoost, Lightgbm, RFRegressor
# Model = XGBoost.XGBoost
# Model = Lightgbm.Lightgbm_sklearn
# Model = Lightgbm.Lightgbm
Model = RFRegressor


submit = False

record = False

test_config = {
    'folds': 5,
    'feature_list': feature_list,
    'model': Model,
    # Best lightgbm param with geo_neighborhood and geo_zip, 0.0646722050526
    # 'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.012923556870735842,
    #     'metric': 'mse', 'min_data': 200, 'min_hessian': 0.232705809294419,
    #     'num_boost_round': 500, 'num_leaves': 30, 'objective': 'regression',
    #     'sub_feature': 0.1596488603529622, 'verbose': -1},
    # Best lightgbm param with almost all features, 0.0646671968399
    # 'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.05525896243523381,
    #     'metric': 'mae', 'min_data': 260, 'min_hessian': 0.57579034653711,
    #     'num_boost_round': 300, 'num_leaves': 70, 'objective': 'regression_l1',
    #     'sub_feature': 0.06638755200543586, 'verbose': -1},
    # rf parameters
    'model_params': {'max_features': 0.2, 'max_leaf_nodes': None,
            'min_samples_leaf': 70, 'n_estimators': 90},
    'submit': submit,
    'record': record,
    # Best xgboost with 5 added features: 0647672681377
    # 'model_params': {'alpha': 0.3, 'colsample_bylevel': 0.3,
    #     'colsample_bytree': 0.5, 'eta': 0.07455450922244707,
    #     'eval_metric': 'mae', 'gamma': 8.249459830776771e-05, 'lambda': 0.6,
    #     'max_depth': 4, 'min_child_weight': 0.9055707037083442,
    #     'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6},
    # 'outliers_lw_pct': 2,
    # 'outliers_up_pct': 98
}
