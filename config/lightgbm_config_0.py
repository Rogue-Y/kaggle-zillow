import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Configuration:
# folds number of K-Fold
FOLDS = 5

# Feature list
from features import lightgbm_feature_list_0
feature_list = lightgbm_feature_list_0.feature_list

# model
from models import Lightgbm
# Model = Lightgbm.Lightgbm_sklearn
Model = Lightgbm.Lightgbm


submit = False

record = False

lightgbm_config_0 = {
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'folds': FOLDS,
    'feature_list': feature_list,
    'model': Model,
    # Best lightgbm param with geo_neighborhood and geo_zip, 0.0646722050526
    'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.012923556870735842,
        'metric': 'mse', 'min_data': 200, 'min_hessian': 0.232705809294419,
        'num_boost_round': 500, 'num_leaves': 30, 'objective': 'regression',
        'sub_feature': 0.1596488603529622, 'verbose': -1},
    'outliers_lw_pct': 0, 'outliers_up_pct': 100,
    'submit': submit,
    'record': record,
}
