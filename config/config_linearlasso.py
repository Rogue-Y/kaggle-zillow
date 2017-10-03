import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_linearlasso
# model
from models import LinearModel
# for defining tunning parameters
from hyperopt import hp

# Configuration
config_linearlasso = {
    'name': 'config_linearlasso',
    'Model': LinearModel.LassoRegressor,
    'feature_list': feature_list_linearlasso.feature_list,
    'clean_na': True,
    'training_params': {
        'model_params': {'alpha': 1.0693486250127264, 'fit_intercept': False, 'normalize': False, 'random_state': 42, 'tol': 0.0025155077434141472},
        'FOLDS': 3,
        'record': False,
        'outliers_up_pct': 98,
        'outliers_lw_pct': 3,
        # 'resale_offset': 0.012
        'pca_components': -1, # clean_na needs to be True to use PCA
        'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'stacking_params': {
        'model_params': {'alpha': 1.0693486250127264, 'fit_intercept': False, 'normalize': False, 'random_state': 42,
                         'tol': 0.0025155077434141472},
        'FOLDS': 3,
        'record': False,
        'outliers_up_pct': 98,
        'outliers_lw_pct': 3,
        # 'resale_offset': 0.012
        'pca_components': -1,  # clean_na needs to be True to use PCA
        'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -2, 2),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'random_state': 42,
                'tol': hp.loguniform('tol', -6, -2),
                'normalize': hp.choice('normalize', [True, False]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'scaling': True,
            'pca_components': hp.choice('pca_components', [-1, 150, 200]),
        },
        'max_evals': 500
    }
}
