import os
import sys
import sklearn
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_linear, feature_list_linearridge, feature_list_linearlasso
# model
from models import LinearModel
# for defining tunning parameters
from hyperopt import hp

# Configuration
config_linearridge = {
    # Focus on Geo
    'name': 'config_linearridge',
    'Model': LinearModel.RidgeRegressor,
    'feature_list': feature_list_linearridge.feature_list,
    'clean_na': True,
    'training_params': {
        'model_params': {'alpha': 7.3876593237849635, 'fit_intercept': True, 'random_state': 42, 'solver': 'auto', 'tol': 0.05789923387169963},
        'outliers_lw_pct': 5, 'outliers_up_pct': 97, 'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'stacking_params': {
        'model_params': {'alpha': 7.3876593237849635, 'fit_intercept': True, 'random_state': 42, 'solver': 'auto', 'tol': 0.05789923387169963},
        'outliers_lw_pct': 5, 'outliers_up_pct': 97, 'scaling': True,
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
        },
        'max_evals': 500
    }
}

# Configuration
config_linear_huber = {
    'name': 'config_linear_huber',
    'Model': LinearModel.Huber,
    'feature_list': feature_list_linear.feature_list,
    'clean_na': True,
    'training_params': {
        'model_params': {
            'alpha': 0.0337035513301721,
            'epsilon': 2.787714318716998,
            'fit_intercept': True,
            'max_iter': 500
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 97,
        'scaling': True
    },
    'stacking_params': {
        'model_params': {'alpha': 0.0337035513301721, 'epsilon': 2.787714318716998, 'fit_intercept': True, 'max_iter': 500}, 'outliers_lw_pct': 5, 'outliers_up_pct': 97, 'scaling': True
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
        },
        'max_evals': 1000
    }
}


# Configuration
config_linearlasso = {
    'name': 'config_linearlasso',
    'Model': LinearModel.LassoRegressor,
    'feature_list': feature_list_linearlasso.feature_list,
    'clean_na': True,
    'training_params': {
        # new data
        {'model_params': {'alpha': 0.20379257521578062, 'fit_intercept': False, 'normalize': True, 'random_state': 42,
                          'tol': 0.017284189969998448},
         'outliers_lw_pct': 4, 'outliers_up_pct': 97, 'scaling': True}
    },
    'stacking_params': {
        {'model_params': {'alpha': 0.20379257521578062, 'fit_intercept': False, 'normalize': True, 'random_state': 42,
                          'tol': 0.017284189969998448},
         'outliers_lw_pct': 4, 'outliers_up_pct': 97, 'scaling': True}

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
            # 'pca_components': hp.choice('pca_components', [-1, 150, 200]),
        },
        'max_evals': 500
    }
}


# Configuration
config_linearRANSAC = {
    'name': 'config_linearRANSAC',
    'Model': LinearModel.RANSAC,
    'feature_list': feature_list_linearridge.feature_list,
    'clean_na': True,
    'training_params': {
        'model_params': {'base_estimator': sklearn.linear_model.Ridge(alpha=7.375287218066115, random_state=42),
                         'min_samples': 0.9697366469576226, 'random_state': 42},
        'record': False,
        'outliers_up_pct': 98,
        'outliers_lw_pct': 5,
        # 'resale_offset': 0.012
        'pca_components': -1,  # clean_na needs to be True to use PCA
        'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'stacking_params': {
        'model_params': {'base_estimator': sklearn.linear_model.Ridge(alpha=7.375287218066115, random_state=42),
                         'min_samples': 0.9697366469576226, 'random_state': 42},
        'outliers_up_pct': 98,
        'outliers_lw_pct': 5,
        # 'resale_offset': 0.012
        'pca_components': -1,  # clean_na needs to be True to use PCA
        'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'base_estimator': sklearn.linear_model.Ridge(alpha=2.22721163144679, random_state=42),
                'min_samples': hp.uniform('min_samples', 0, 1),
                'random_state': 42,
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'scaling': True,
            'pca_components': -1,
        },
        'max_evals': 100
    }
}

config_linearRANSAC2 = {
    'name': 'config_linearRANSAC2',
    'Model': LinearModel.RANSAC,
    'feature_list': feature_list_linearridge.feature_list2,
    'clean_na': True,
    'training_params': {
        'model_params': {'base_estimator': sklearn.linear_model.Ridge(alpha=2.22721163144679, random_state=42),
                         'min_samples': 0.6305959242246917, 'random_state': 42},
        'record': False,
        'outliers_up_pct': 97,
        'outliers_lw_pct': 4,
        # 'resale_offset': 0.012
        'pca_components': -1,  # clean_na needs to be True to use PCA
        'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'stacking_params': {
        'model_params': {'base_estimator': sklearn.linear_model.Ridge(alpha=2.22721163144679, random_state=42),
                         'min_samples': 0.9697366469576226, 'random_state': 42},
        'record': False,
        'outliers_up_pct': 97,
        'outliers_lw_pct': 4,
        # 'resale_offset': 0.012
        'pca_components': -1,  # clean_na needs to be True to use PCA
        'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'base_estimator': sklearn.linear_model.Ridge(alpha=2.22721163144679, random_state=42),
                'min_samples': hp.uniform('min_samples', 0, 1),
                'random_state': 42,
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'scaling': True,
            'pca_components': -1,
        },
        'max_evals': 100
    }
}
