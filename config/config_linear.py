import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_linear
# model
from models import LinearModel
# for defining tunning parameters
from hyperopt import hp

# Configuration
config_linear = {
    'name': 'config_linear',
    'Model': LinearModel.RidgeRegressor,
    'feature_list': feature_list_linear.feature_list,
    'clean_na': True,
    'training_params': {
        'model_params': {'alpha': 1.0, 'random_state': 42},
        'FOLDS': 5,
        'record': False,
        'outliers_up_pct': 99,
        'outliers_lw_pct': 1,
        # 'resale_offset': 0.012
        'pca_components': -1, # clean_na needs to be True to use PCA
        'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'stacking_params': {
        'model_params': {'alpha': 1.0, 'random_state': 42},
        'FOLDS': 2,
        'outliers_up_pct': 99,
        'outliers_lw_pct': 1,
        'pca_components': -1, # clean_na needs to be True to use PCA
        'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -2, 2),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
                'random_state': 42
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'scaling': True,
            'pca_components': hp.choice('pca_components', [-1, 30, 50, 100, 150, 200]),
        },
        'max_evals': 500
    }
}
