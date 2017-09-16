import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Configuration:
# folds number of K-Fold
FOLDS = 5

# Feature list
from features import xgb_feature_list_2
feature_list = xgb_feature_list_2.feature_list

# model
from models import XGBoost
Model = XGBoost.XGBoost

submit = False

record = False

xgb_config_2 = {
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'folds': FOLDS,
    'feature_list': feature_list,
    'model': Model,
    'submit': submit,
    'record': record,

    # xgboost with 5 added features + target features: 0.0644820716823
    'model_params': {'alpha': 0.0, 'colsample_bylevel': 0.5,
        'colsample_bytree': 0.6, 'eta': 0.057797411022032265,
        'eval_metric': 'rmse', 'gamma': 0.026493673908889032, 'lambda': 0.7,
        'max_depth': 6, 'min_child_weight': 1.913900160701277,
        'objective': 'reg:linear', 'silent': 1, 'subsample': 0.8},
    'outliers_lw_pct': 4, 'outliers_up_pct': 97
}
