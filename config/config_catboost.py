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
config_catboost = {
    'name': 'config_catboost',
    'Model': CatBoost.CatBoost,
    'feature_list': feature_list_cat.feature_list,
    'clean_na': False,
    'training_params': {
        # New Full data
        'model_params': {'depth': 5, 'eval_metric': 'MAE', 'iterations': 200, 'l2_leaf_reg': 4, 'learning_rate': 0.018995235001009962,
                         'loss_function': 'MAE', 'random_seed': 42},
        # 'record': False,
        'outliers_lw_pct': 4,
        'outliers_up_pct': 100,
    },
    'stacking_params': {
        'model_params': {'iterations':200, 'learning_rate':0.026546125048271585, 'depth':7, 'l2_leaf_reg':3, 'loss_function':'MAE',
                         'eval_metric':'MAE', 'random_seed':42},
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
                'iterations': hp.choice('iterations', [200, 300]),
                'learning_rate': hp.loguniform('learning_rate', -4, -2),
                'depth': hp.choice('depth', list(range(4, 8))),
                'l2_leaf_reg': hp.choice('l2_leaf_reg', list(range(2, 6))),
                'loss_function': 'MAE',
                'eval_metric' : 'MAE',
                'random_seed' : 42,
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [2, 1, 0]),
        },
        'max_evals': 1
    }
}

config_manycatsboost = {
    'name': 'config_manycatsboost',
    'Model': CatBoost.ManyCatsBoost,
    'feature_list': feature_list_cat.feature_list,
    'clean_na': False,
    'training_params': {
                'model_params': {'depth': 5, 'eval_metric': 'MAE', 'iterations': 200, 'l2_leaf_reg': 4, 'learning_rate': 0.018995235001009962,
                         'loss_function': 'MAE'},
        # 'model_params': {'iterations': 300, 'learning_rate': 0.021788752145849327, 'depth': 6, 'l2_leaf_reg': 3,
        #                  'loss_function': 'MAE',
        #                  'eval_metric': 'MAE'},
        # 'record': False,
        'outliers_lw_pct': 0,
        'outliers_up_pct': 100,
        # 'resale_offset': 0.012
        # 'pca_components': -1, # clean_na needs to be True to use PCA
        # 'scaling': False,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'stacking_params': {
        'model_params': {'iterations': 200, 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 3, 'loss_function': 'MAE',
                         'eval_metric': 'MAE'},
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
                # 'eta': hp.loguniform('eta', -2, 0),
                # 'gamma': hp.loguniform('gamma', -4, -1),
                # 'max_depth': hp.choice('max_depth', list(range(2, 7))),
                # 'min_child_weight': hp.uniform('min_child_weight', 3, 6),
                # 'subsample': hp.choice('subsample', [x/10 for x in range(5, 10)]),
                # 'colsample_bytree': hp.choice('colsample_bytree', [x/10 for x in range(5, 11)]),
                # 'colsample_bylevel': hp.choice('colsample_bylevel', [x/10 for x in range(5, 11)]),
                # 'lambda': hp.choice('lambda', [x/10 for x in range(0, 6)]),
                # 'alpha': hp.choice('alpha', [x/10 for x in range(3, 8)]),
                # 'objective': 'reg:linear',
                # 'eval_metric': hp.choice('eval_metric', ['mae', 'rmse']),
                # # 'base_score': y_mean,
                # # 'booster': 'gblinear',
                # 'silent': 1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
        },
        'max_evals': 100
    }
}
