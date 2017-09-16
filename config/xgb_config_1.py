import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Configuration:
# folds number of K-Fold
FOLDS = 5

# Feature list
from features import xgb_feature_list_1
feature_list = xgb_feature_list_1.feature_list

# model
from models import XGBoost
Model = XGBoost.XGBoost

submit = False

record = False

xgb_config_1 = {
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'folds': FOLDS,
    'feature_list': feature_list,
    'model': Model,
    'submit': submit,
    'record': record,

    # Best xgboost with 5 added features + precise geo filling: 0647672681377, lb: 0.0644668
    'model_params': {'alpha': 0.3, 'colsample_bylevel': 0.3,
        'colsample_bytree': 0.5, 'eta': 0.07455450922244707,
        'eval_metric': 'mae', 'gamma': 8.249459830776771e-05, 'lambda': 0.6,
        'max_depth': 4, 'min_child_weight': 0.9055707037083442,
        'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6},
    'outliers_lw_pct': 2,
    'outliers_up_pct': 98
}
