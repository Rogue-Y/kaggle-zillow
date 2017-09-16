import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Configuration:
# folds number of K-Fold
FOLDS = 5

# Feature list
from features import lightgbm_feature_list_1
feature_list = lightgbm_feature_list_1.feature_list

# model
from models import Lightgbm
# Model = Lightgbm.Lightgbm_sklearn
Model = Lightgbm.Lightgbm


submit = False

record = False

lightgbm_config_1 = {
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'folds': FOLDS,
    'feature_list': feature_list,
    'model': Model,
    # Best lightgbm param with almost all features, 0.0646671968399
    'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.05525896243523381,
        'metric': 'mae', 'min_data': 260, 'min_hessian': 0.57579034653711,
        'num_boost_round': 300, 'num_leaves': 70, 'objective': 'regression_l1',
        'sub_feature': 0.06638755200543586, 'verbose': -1},
    'outliers_lw_pct': 0, 'outliers_up_pct': 100,
    'submit': submit,
    'record': record,
}
