from models import Lightgbm, LinearModel
from .config_lightgbm import config_lightgbm, config_lightgbm_new, config_lightgbm_geo
from .config_xgboost import config_xgboost, config_xgboost_new
from .config_ensembles import config_rf, config_extra_tree, config_gb
from .config_linear import config_linear_huber, config_linearlasso, config_linearridge, config_linearRANSAC

from hyperopt import hp

stacking_config_test = {
    'name': 'stacking_config_test',
    'stacking_list': [
        (config_linearRANSAC, False),
        (config_linearridge, False),
        (config_linear_huber, False),
        (config_linearlasso, False),
        (config_lightgbm, False),
        (config_lightgbm_geo, False),
        (config_xgboost, False),
        (config_rf, False),
        (config_extra_tree, False),
        (config_gb, False),
    ],
    'global_force_generate': False,
    'Meta_model': Lightgbm.Lightgbm,
    # predicting parameters
    'model_params': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.14073826510900514,
        'max_bin': 50,
        'metric': 'mse',
        'min_data': 300,
        'min_hessian': 1.776106930543375,
        'num_boost_round': 200,
        'num_leaves': 10,
        'objective': 'regression',
        'verbose': -1
    },
    'outliers_lw_pct': 5,
    'outliers_up_pct': 95,
    # 'resale_offset': 0.012,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, 0),
                'boosting_type': 'gbdt',
                'objective': hp.choice('objective', ['regression_l2', 'regression_l1', 'huber', 'fair', 'poisson']),
                'metric': hp.choice('metric', ['mae', 'mse']),
                'sub_feature': hp.choice('sub_feature', [0.5, 0.6, 0.7, 0.8, 0.9, 1]),
                'num_leaves': hp.choice('num_leaves', list(range(5, 101, 10))),
                'min_data': hp.choice('min_data', list(range(200, 500, 15))),
                'min_hessian': hp.loguniform('min_hessian', -1, 1),
                'num_boost_round': hp.choice('num_boost_round', [200, 300, 500, 1000]),
                'max_bin': hp.choice('max_bin', list(range(10, 251, 20))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [94, 95, 96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [6, 5, 4, 3, 2, 1, 0]),
        },
        'max_evals': 100000,
    }
}


stacking_config_linear = {
    'name': 'stacking_config_linear',
    'stacking_list': [
        (config_lightgbm, False),
        (config_lightgbm_geo, False),
        (config_linearridge, False),
        (config_linear_huber, False),
        (config_linearlasso, False),
        (config_linearRANSAC, False),
        (config_xgboost, False),
        (config_rf, False),
        (config_extra_tree, False),
        (config_gb, False),
    ],
    'global_force_generate': False,
    'Meta_model': LinearModel.Linear,
    # predicting parameters
    # cv: 0.0646115110435665
    'model_params': {
        'fit_intercept': True,
        'normalize': True
    },
    'outliers_lw_pct': 4,
    'outliers_up_pct': 95,
    # 'resale_offset': 0.012,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'normalize': hp.choice('normalize', [True, False]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [94, 95, 96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [6, 5, 4, 3, 2, 1, 0]),
        },
        'max_evals': 200
    }
}


stacking_config_ridge = {
    'name': 'stacking_config_ridge',
    'stacking_list': [
        (config_lightgbm, False),
        (config_lightgbm_geo, False),
        (config_linearridge, False),
        (config_linear_huber, False),
        (config_linearlasso, False),
        (config_linearRANSAC, False),
        (config_xgboost, False),
        (config_rf, False),
        (config_extra_tree, False),
        (config_gb, False),
    ],
    'global_force_generate': False,
    'Meta_model': LinearModel.RidgeRegressor,
    # predicting parameters
    'model_params': {
        'alpha': 0.1503527992652327,
        'fit_intercept': False,
        'random_state': 42,
        'solver': 'sag',
        'tol': 0.04893417145299664
    },
    'outliers_lw_pct': 1,
    'outliers_up_pct': 98,
    # 'resale_offset': 0.012,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -3, 2),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
                'tol': hp.loguniform('tol', -6, -2),
                'random_state': 42
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [94, 95, 96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [6, 5, 4, 3, 2, 1, 0]),
        },
        'max_evals': 30000
    }
}
