import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_non_linear, feature_list_feature_importance
# model
from models import XGBoost
# for tunning parameters
from hyperopt import hp

# Configuration
config_xgboost = {
    'name': 'config_xgboost',
    'Model': XGBoost.XGBoost,
    'feature_list': feature_list_non_linear.feature_list,
    'clean_na': False,
    'training_params': {
        # 'model_params': {'alpha': 0.6, 'colsample_bylevel': 0.7, 'colsample_bytree': 0.7, 'eta': 0.07901772316032044,
        #     'eval_metric': 'rmse', 'gamma': 0.0018188912716341973, 'lambda': 0.4, 'max_depth': 4,
        #     'min_child_weight': 4.4156043204121, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6},
        # 'FOLDS': 2,
        # # 'record': False,
        # 'outliers_lw_pct': 4,
        # 'outliers_up_pct': 97,
        # # 'resale_offset': 0.012
        # # 'pca_components': -1, # clean_na needs to be True to use PCA
        # # 'scaling': False,
        # # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # # 'scaling_columns': SCALING_COLUMNS

        'model_params': {
            'alpha': 0.3,
            'colsample_bylevel': 1.0,
            'colsample_bytree': 0.2,
            'eta': 0.23350979706978167,
            'eval_metric': 'mae',
            'gamma': 0.04379710673998437,
            'lambda': 0.4,
            'max_depth': 1,
            'min_child_weight': 3.193139219626233,
            'objective': 'reg:linear',
            'silent': 1,
            'subsample': 0.9
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 96
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'eta': hp.loguniform('eta', -2, -1),
                'gamma': hp.loguniform('gamma', -4, -1),
                'max_depth': hp.choice('max_depth', list(range(2, 7))),
                'min_child_weight': hp.uniform('min_child_weight', 3, 6),
                'subsample': hp.choice('subsample', [x/10 for x in range(5, 10)]),
                'colsample_bytree': hp.choice('colsample_bytree', [x/10 for x in range(5, 11)]),
                'colsample_bylevel': hp.choice('colsample_bylevel', [x/10 for x in range(5, 11)]),
                'lambda': hp.choice('lambda', [x/10 for x in range(0, 6)]),
                'alpha': hp.choice('alpha', [x/10 for x in range(3, 8)]),
                'objective': 'reg:linear',
                'eval_metric': hp.choice('eval_metric', ['mae', 'rmse']),
                # 'base_score': y_mean,
                # 'booster': 'gblinear',
                'silent': 1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
        },
        'max_evals': 100
    }
}

# Configuration
config_xgboost_fi = {
    'name': 'config_xgboost_fi',
    'Model': XGBoost.XGBoost,
    'feature_list': feature_list_feature_importance.feature_list,
    'clean_na': False,
    'training_params': {
        'model_params': {
            'alpha': 0.5,
            'colsample_bylevel': 0.5,
            'colsample_bytree': 0.7,
            'eta': 0.1498378171464992,
            'eval_metric': 'rmse',
            'gamma': 0.028430685832544367,
            'lambda': 0.2,
            'max_depth': 2,
            'min_child_weight': 5.7702332347547785,
            'objective': 'reg:linear',
            'silent': 1,
            'subsample': 0.8
        },
        'outliers_lw_pct': 3,
        'outliers_up_pct': 97
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'eta': hp.loguniform('eta', -2, -1),
                'gamma': hp.loguniform('gamma', -3, -2),
                'max_depth': hp.choice('max_depth', list(range(2, 8))),
                'min_child_weight': hp.uniform('min_child_weight', 3.5, 6.5),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'colsample_bytree': hp.uniform('subsample', 0.5, 1),
                # 'colsample_bylevel': hp.choice('colsample_bylevel', [x/10 for x in range(5, 11)]),
                'lambda': hp.uniform('lambda', 0.1, 1),
                # 'alpha': hp.uniform('alpha', [x/10 for x in range(3, 8)]),
                'objective': 'reg:linear',
                'eval_metric': hp.choice('eval_metric', ['mae', 'rmse']),
                # 'base_score': y_mean,
                # 'booster': 'gblinear',
                'silent': 1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
        },
        'max_evals': 250
    },
}

# Configuration
config_xgboost_all = {
    'name': 'config_xgboost_all',
    'Model': XGBoost.XGBoost,
    'feature_list': feature_list_non_linear.feature_list_all,
    'clean_na': False,
    'training_params': {
    },
    'stacking_params': {
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'eta': hp.loguniform('eta', -2, -1),
                'gamma': hp.loguniform('gamma', -3, -2),
                'max_depth': hp.choice('max_depth', list(range(2, 8))),
                'min_child_weight': hp.uniform('min_child_weight', 3.5, 6.5),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                # 'colsample_bylevel': hp.choice('colsample_bylevel', [x/10 for x in range(5, 11)]),
                'lambda': hp.uniform('lambda', 0.1, 1),
                # 'alpha': hp.uniform('alpha', [x/10 for x in range(3, 8)]),
                'objective': 'reg:linear',
                'eval_metric': hp.choice('eval_metric', ['mae', 'rmse']),
                # 'base_score': y_mean,
                # 'booster': 'gblinear',
                'silent': 1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [99, 98, 97, 96, 95]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [1, 2, 3, 4, 5]),
        },
        'max_evals': 250
    },
}

# Configuration
config_xgboost_all_dart = {
    'name': 'config_xgboost_all_dart',
    'Model': XGBoost.XGBoost,
    'feature_list': feature_list_non_linear.feature_list_all,
    'clean_na': False,
    'training_params': {
        'model_params': {
             'booster': 'dart',
             'max_depth': 1, 'learning_rate': 0.1,
             'objective': 'reg:linear', 'silent': True,
             'sample_type': 'uniform',
             'normalize_type': 'tree',
             'rate_drop': 0.1,
             'skip_drop': 0.5
            }
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'booster': 'dart',
                # 'sample_type': 'uniform',
                # 'normalize_type': 'tree',
                'rate_drop': hp.uniform('rate_drop', 0.1, 0.2),
                'skip_drop': hp.choice('skip_drop', [0.3, 0.5, 0.7]),
                'eta': hp.loguniform('eta', -2, -1),
                'gamma': hp.loguniform('gamma', -3, -2),
                'max_depth': hp.choice('max_depth', list(range(2, 8))),
                'min_child_weight': hp.uniform('min_child_weight', 3.5, 6.5),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.05, 0.7),
                # 'colsample_bylevel': hp.choice('colsample_bylevel', [x/10 for x in range(5, 11)]),
                'lambda': hp.uniform('lambda', 0.1, 1),
                # 'alpha': hp.uniform('alpha', [x/10 for x in range(3, 8)]),
                'objective': 'reg:linear',
                'eval_metric': hp.choice('eval_metric', ['mae', 'rmse']),
                # 'base_score': y_mean,
                # 'booster': 'gblinear',
                'silent': 1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [99, 98, 97, 96, 95]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [1, 2, 3, 4, 5]),
        },
        'max_evals': 300
    },
}


# Configuration
config_xgboost_new = {
    'name': 'config_xgboost_new',
    'Model': XGBoost.XGBoost,
    'feature_list': feature_list_non_linear.feature_list,
    'clean_na': False,
    'training_params': {
        'FOLDS': 3,
        'model_params': {
            'alpha': 0.5,
            'colsample_bylevel': 0.4,
            'colsample_bytree': 0.8,
            'eta': 0.19071472448799803,
            'eval_metric': 'rmse',
            'gamma': 0.05504791294083944,
            'lambda': 0.2,
            'max_depth': 3,
            'min_child_weight': 5.3013678371644675,
            'objective': 'reg:linear',
            'silent': 1,
            'subsample': 0.7
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 97
    },
    'stacking_params': {
        'FOLDS': 3,
        'model_params': {
            'alpha': 0.5,
            'colsample_bylevel': 0.4,
            'colsample_bytree': 0.8,
            'eta': 0.19071472448799803,
            'eval_metric': 'rmse',
            'gamma': 0.05504791294083944,
            'lambda': 0.2,
            'max_depth': 3,
            'min_child_weight': 5.3013678371644675,
            'objective': 'reg:linear',
            'silent': 1,
            'subsample': 0.7
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 97
    }
}
