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
from models import XGBoost, Lightgbm
#Model = XGBoost.XGBoost
# Model = Lightgbm.Lightgbm_sklearn
Model = Lightgbm.Lightgbm

submit = True

record = False

test_config = {
    'folds': 5,
    'feature_list': feature_list,
    'model': Model,
    # Best lightgbm param with geo_neighborhood and geo_zip
    'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.012923556870735842,
        'metric': 'mse', 'min_data': 200, 'min_hessian': 0.232705809294419,
        'num_boost_round': 500, 'num_leaves': 30, 'objective': 'regression',
        'sub_feature': 0.1596488603529622, 'verbose': -1},
    'submit': submit,
    'record': record
}
