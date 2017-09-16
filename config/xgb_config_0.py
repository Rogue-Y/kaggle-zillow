import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Configuration:
# folds number of K-Fold
FOLDS = 5

# Feature list
from features import xgb_feature_list_0
feature_list = xgb_feature_list_0.feature_list

# model
from models import XGBoost
Model = XGBoost.XGBoost

submit = False

record = False

xgb_config_0 = {
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'folds': FOLDS,
    'feature_list': feature_list,
    'model': Model,
    'submit': submit,
    'record': record,

    # further tuning of the above one: cv 0.0646223502383, lb: 0.0643487
    # changed order of features (put missing values first): cv 0.0647246470924, lb: 0.0643368
    # plus target: cv: 0.0645315564755 lb: 0.0643622
    'model_params': {'alpha': 0.6, 'colsample_bylevel': 0.7, 'colsample_bytree': 0.7, 'eta': 0.07901772316032044,
        'eval_metric': 'rmse', 'gamma': 0.0018188912716341973, 'lambda': 0.4, 'max_depth': 4,
        'min_child_weight': 4.4156043204121, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6},
    'outliers_lw_pct': 4, 'outliers_up_pct': 97
}
