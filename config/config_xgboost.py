import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_non_linear

# model
from models import XGBoost

# Configuration
config_xgboost = {
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'feature_list': feature_list_non_linear.feature_list,
    'clean_na': False,
    'training_params': {
        'Model': XGBoost.XGBoost,
        'model_params': {'alpha': 0.6, 'colsample_bylevel': 0.7, 'colsample_bytree': 0.7, 'eta': 0.07901772316032044,
            'eval_metric': 'rmse', 'gamma': 0.0018188912716341973, 'lambda': 0.4, 'max_depth': 4,
            'min_child_weight': 4.4156043204121, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6},
        'FOLDS': 2,
        # 'record': False,
        'outliers_lw_pct': 4,
        'outliers_up_pct': 97,
        # 'resale_offset': 0.012
        # 'pca_components': -1, # clean_na needs to be True to use PCA
        # 'scaling': False,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'stacking_params': {
        'Model': XGBoost.XGBoost,
        'model_params': {'alpha': 0.6, 'colsample_bylevel': 0.7, 'colsample_bytree': 0.7, 'eta': 0.07901772316032044,
            'eval_metric': 'rmse', 'gamma': 0.0018188912716341973, 'lambda': 0.4, 'max_depth': 4,
            'min_child_weight': 4.4156043204121, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6},
        'FOLDS': 5,
        'outliers_lw_pct': 4,
        'outliers_up_pct': 97,
        # 'pca_components': -1, # clean_na needs to be True to use PCA
        # 'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    }
}
