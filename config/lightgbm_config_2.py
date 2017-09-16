import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Configuration:
# folds number of K-Fold
FOLDS = 5

# Feature list
from features import lightgbm_feature_list_2
feature_list = lightgbm_feature_list_2.feature_list

# model
from models import Lightgbm
# Model = Lightgbm.Lightgbm_sklearn
Model = Lightgbm.Lightgbm


submit = False

record = False

lightgbm_config_2 = {
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'folds': FOLDS,
    'feature_list': feature_list,
    'model': Model,
    'submit': submit,
    'record': record,

    # lightgbm: cv 0.064637416215, lb:0.0645447
    'model_params': {'bagging_fraction': 0.8384638110940468, 'bagging_freq': 0,
        'boosting_type': 'gbdt', 'learning_rate': 0.1353711356306096,
        'max_bin': 100, 'metric': 'mse', 'min_data': 230,
        'min_hessian': 0.5961775594444781, 'num_boost_round': 200, 'num_leaves': 50,
        'objective': 'regression', 'sub_feature': 0.18462105358643505, 'verbose': -1},
    'outliers_lw_pct': 4, 'outliers_up_pct': 96
}
