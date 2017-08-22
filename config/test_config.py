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
Model = Lightgbm.Lightgbm_sklearn

submit = False

test_config = {
    'folds': 5,
    'feature_list': feature_list,
    'model': Model,
    'submit': False
}
