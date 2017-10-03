import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_non_linear
# model
from models import Ensemble
# for defining tunning parameters
from hyperopt import hp

# Random forest
config_rf = {
    'name': 'config_rf',
    'Model': Ensemble.RandomForest,
    'feature_list': feature_list_non_linear.feature_list,
    'clean_na': True,
    'training_params': {
        'FOLDS': 3,
        'model_params': {
            'criterion': 'mse',
            'max_depth': 5,
            'max_features': 0.29128611952923245,
            'min_samples_leaf': 0.01834965011541529,
            'min_samples_split': 0.025518924298922253,
            'n_estimators': 90,
            'n_jobs': -1
        },
        'outliers_lw_pct': 4,
        'outliers_up_pct': 97
    },
    'stacking_params': {
        'FOLDS': 3,
        'model_params': {
            'criterion': 'mse',
            'max_depth': 5,
            'max_features': 0.29128611952923245,
            'min_samples_leaf': 0.01834965011541529,
            'min_samples_split': 0.025518924298922253,
            'n_estimators': 90,
            'n_jobs': -1
        },
        'outliers_lw_pct': 4,
        'outliers_up_pct': 97
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'n_estimators': hp.choice('n_estimators', list(range(30, 101, 10))),
                'criterion': 'mse',
                'max_features': hp.loguniform('max_features', -2, -1),
                'max_depth': hp.choice('max_depth', list(range(3, 6))),
                'min_samples_split': hp.loguniform('min_samples_split', -4, -2),
                'min_samples_leaf': hp.loguniform('min_samples_leaf', -4, -2),
                'n_jobs': -1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'FOLDS': 3 #RF takes long time to train
        },
        'max_evals': 75
    }
}


# Extra tree
config_extra_tree = {
    'name': 'config_extra_tree',
    'Model': Ensemble.ExtraTrees,
    'feature_list': feature_list_non_linear.feature_list,
    'clean_na': True,
    'training_params': {
        'FOLDS': 3,
        'model_params': {
            'criterion': 'mse',
            'max_depth': 7,
            'max_features': 0.5527621519513952,
            'min_samples_leaf': 0.0186596367173352,
            'min_samples_split': 0.029402685599045443,
            'n_estimators': 40,
            'n_jobs': -1
        },
        'outliers_lw_pct': 4,
        'outliers_up_pct': 97
    },
    'stacking_params': {
        'FOLDS': 3,
        'model_params': {
            'criterion': 'mse',
            'max_depth': 7,
            'max_features': 0.5527621519513952,
            'min_samples_leaf': 0.0186596367173352,
            'min_samples_split': 0.029402685599045443,
            'n_estimators': 40,
            'n_jobs': -1
        },
        'outliers_lw_pct': 4,
        'outliers_up_pct': 97
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'n_estimators': hp.choice('n_estimators', list(range(30, 51, 10))),
                'criterion': 'mse',
                'max_features': hp.uniform('max_features', 0.1, 0.6),
                'max_depth': hp.choice('max_depth', list(range(3, 8, 2))),
                'min_samples_split': hp.loguniform('min_samples_split', -4, -1),
                'min_samples_leaf': hp.loguniform('min_samples_leaf', -4, -2),
                # 'bootstrap': hp.choice('bootstrap', [True, False]),
                # 'oob_score': hp.choice('oob_score', [True, False]),
                'n_jobs': -1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'FOLDS': 3 #RF takes long time to train
        },
        'max_evals': 75
    }
}


# Gradient boosting
config_gb = {
    'name': 'config_gb',
    'Model': Ensemble.GradientBoosting,
    'feature_list': feature_list_non_linear.feature_list,
    'clean_na': True,
    'training_params': {
        'FOLDS': 3,
        'model_params': {
            'alpha': 0.9,
            'criterion': 'friedman_mse',
            'learning_rate': 0.22394116155015484,
            'loss': 'ls',
            'max_depth': 6,
            'max_features': 0.3098657218048529,
            'min_samples_leaf': 0.024563708341208284,
            'min_samples_split': 0.0985392674346176,
            'n_estimators': 300,
            'subsample': 1
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 96
    },
    'stacking_params': {
        'FOLDS': 3,
        'model_params': {
            'criterion': 'friedman_mse',
            'learning_rate': 0.17793166182438155,
            'loss': 'ls',
            'max_depth': 6,
            'max_features': 0.26065972420384176,
            'min_samples_leaf': 0.08832202126396065,
            'min_samples_split': 0.023728156535801175,
            'n_estimators': 300,
            'subsample': 1
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 96
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'loss': hp.choice('loss', ['ls', 'lad', 'huber', 'quantile']),
                'learning_rate': hp.loguniform('learning_rate', -2, 0),
                'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200, 300]),
                'max_depth': hp.choice('max_depth', list(range(2, 7))),
                'criterion': hp.choice('criterion', ['mse', 'friedman_mse']),
                'min_samples_split': hp.loguniform('min_samples_split', -4, -2),
                'min_samples_leaf': hp.loguniform('min_samples_leaf', -4, -2),
                'subsample': hp.choice('subsample', [0.3, 0.5, 0.8, 1]),
                'max_features': hp.loguniform('max_features', -2, -1),
                'alpha': hp.choice('alpha', [0.3, 0.6, 0.9]), # only for hubor and quantile loss                # 'bootstrap': hp.choice('bootstrap', [True, False]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'FOLDS': 3 #RF takes long time to train
        },
        'max_evals': 200
    }
}
