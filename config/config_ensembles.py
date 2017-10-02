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
        # 'FOLDS': 3,
        # 'model_params': {
        #     'criterion': 'mse',
        #     'max_depth': 3,
        #     'max_features': 0.35373046429724236,
        #     'min_samples_leaf': 0.029133482031154883,
        #     'min_samples_split': 0.08476967706841676,
        #     'n_estimators': 10,
        #     'n_jobs': -1
        # },
        # 'outliers_lw_pct': 4,
        # 'outliers_up_pct': 99,
    
        'FOLDS': 3, 'model_params': {'criterion': 'mse', 'max_depth': 5, 'max_features': 0.29128611952923245, 'min_samples_leaf': 0.01834965011541529, 'min_samples_split': 0.025518924298922253, 'n_estimators': 90, 'n_jobs': -1}, 'outliers_lw_pct': 4, 'outliers_up_pct': 97
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
        'FOLDS': 3, 'model_params': {'criterion': 'mse', 'max_depth': 7, 'max_features': 0.5527621519513952, 'min_samples_leaf': 0.0186596367173352, 'min_samples_split': 0.029402685599045443, 'n_estimators': 40, 'n_jobs': -1}, 'outliers_lw_pct': 4, 'outliers_up_pct': 97
    },
    'stacking_params': {
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
    },
    'stacking_params': {
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
        'max_evals': 75
    }
}
