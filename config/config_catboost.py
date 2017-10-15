import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_cat, feature_list_non_linear
# model
# pip install catboost
from models import CatBoost
# for tunning parameters
from hyperopt import hp

# Configuration
config_catboost_clean = {
    'name': 'config_catboost_clean',
    'Model': CatBoost.CatBoost,
    'feature_list': feature_list_cat.feature_list2,
    'clean_na': True,
    'training_params': {
        # New Full data
        'model_params':  {'depth': 4, 'eval_metric': 'MAE', 'iterations': 300, 'l2_leaf_reg': 4, 'learning_rate': 0.023875439091624318, 'loss_function': 'MAE', 'random_seed': 42},
        # 'record': False,
        'outliers_lw_pct': 0,
        'outliers_up_pct': 100,
    },
    'stacking_params': {
        'model_params':  {'depth': 4, 'eval_metric': 'MAE', 'iterations': 300, 'l2_leaf_reg': 4, 'learning_rate': 0.023875439091624318, 'loss_function': 'MAE', 'random_seed': 42},
        # 'record': False,
        'outliers_lw_pct': 0,
        'outliers_up_pct': 100,
        # 'resale_offset': 0.012
        # 'pca_components': -1, # clean_na needs to be True to use PCA
        # 'scaling': False,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'iterations': hp.choice('iterations', [200, 250, 300]),
                'learning_rate': hp.loguniform('learning_rate', -4, -2),
                'depth': hp.choice('depth', list(range(4, 8))),
                'l2_leaf_reg': hp.choice('l2_leaf_reg', list(range(3, 6))),
                'loss_function': 'MAE',
                'eval_metric' : 'MAE',
                'random_seed' : 42,
            },
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [0, 1, 2, 3, 4]),
            'outliers_up_pct': hp.choice('outliers_up_pct', [96, 97, 98, 99, 100]),
        },
        'max_evals': 30
    }
}

config_manycatsboost_clean = {
    'name': 'config_manycatsboost_clean',
    'Model': CatBoost.ManyCatsBoost,
    'feature_list': feature_list_cat.feature_list,
    'clean_na': False,
    'training_params': {
        # New Full data
        'model_params':  {'depth': 4, 'eval_metric': 'MAE', 'iterations': 300, 'l2_leaf_reg': 4, 'learning_rate': 0.023875439091624318, 'loss_function': 'MAE'},
        # 'record': False,
        'outliers_lw_pct': 0,
        'outliers_up_pct': 100,
    },
    'stacking_params': {
        # New Full data
        'model_params': {'depth': 4, 'eval_metric': 'MAE', 'iterations': 300, 'l2_leaf_reg': 4,
                         'learning_rate': 0.023875439091624318, 'loss_function': 'MAE'},
        # 'record': False,
        'outliers_lw_pct': 0,
        'outliers_up_pct': 100,
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                # Manycatsboost is just 5 normal catboosts averaged.
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
        },
        'max_evals': 100
    }
}

