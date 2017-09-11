# lightgbm config
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
from models import Lightgbm
# Model = Lightgbm.Lightgbm_sklearn
Model = Lightgbm.Lightgbm

submit = False

record = False

lightgbm_config = {
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'folds': FOLDS,
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
    'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.2100954943925603, 
    	'max_bin': 255, 'metric': 'mse', 'min_data': 225, 'min_hessian': 0.06297429722636191,
    	'num_leaves': 10, 'objective': 'regression', 'sub_feature': 0.13114357843072696, 'verbose': -1},
    'outliers_lw_pct': 4,
    'outliers_up_pct': 97, 
    
    'submit': submit,
    'record': record,
}
