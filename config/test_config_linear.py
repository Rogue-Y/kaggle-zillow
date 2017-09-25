import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Configuration:
# folds number of K-Fold
FOLDS = 5

# Feature list
from features import test_feature_list_linear
feature_list = test_feature_list_linear.feature_list

# model
from models import LinearModel
Model = LinearModel.RidgeRegressor


submit = False

record = False

test_config_linear = {
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'folds': FOLDS,
    'scaling': True,
    'feature_list': feature_list,
    'model': Model,
    'model_params': {'alpha': 1.0, 'random_state': 42},
    'clean_na': True,
    'submit': submit,
    'record': record,
}
