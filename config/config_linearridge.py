import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_linearridge
# model
from models import LinearModel
# for defining tunning parameters
from hyperopt import hp

# Configuration
config_linearridge = {
    'name': 'config_linearridge',
    'Model': LinearModel.RidgeRegressor,
    'feature_list': feature_list_linearridge.feature_list2,
    'clean_na': True,
    'training_params': {
        'model_params': {'alpha': 7.375287218066115, 'random_state': 42, 'tol': 0.001},
        'FOLDS': 3,
        'record': False,
        'outliers_up_pct': 97,
        'outliers_lw_pct': 5,
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
                'tol': hp.loguniform('tol', -6, -2),
                'random_state': 42
            },
            'outliers_up_pct': 97, # hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': 5,  #hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'scaling': True,
            'pca_components': -1, #hp.choice('pca_components', [-1, 30, 50, 100, 150, 200]),
        },
        'max_evals': 500
    }
}


# Configuration
config_linear_huber = {
    'name': 'config_linear_huber',
    'Model': LinearModel.Huber,
    'feature_list': feature_list_linearridge.feature_list,
    'clean_na': True,
    'training_params': {
        'FOLDS': 3, 'model_params': {'alpha': 0.021419478306822514, 'epsilon': 3.55839417191035, 'fit_intercept': False, 'max_iter': 500}, 'outliers_lw_pct': 4, 'outliers_up_pct': 98, 'scaling': True
    },
    'stacking_params': {
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'epsilon': hp.loguniform('epsilon', 0, 2),
                'max_iter': hp.choice('max_iter', [50, 100, 200, 500]),
                'alpha': hp.loguniform('alpha', -5, -1),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'scaling': True,
            'FOLDS': 3,
        },
        'max_evals': 50
    }
}
