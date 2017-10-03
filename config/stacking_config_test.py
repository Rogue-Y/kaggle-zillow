from models import Lightgbm
from .config_lightgbm import config_lightgbm, config_lightgbm_new
from .config_xgboost import config_xgboost, config_xgboost_new
from .config_ensembles import config_rf, config_extra_tree, config_gb
from hyperopt import hp

stacking_config_test = {
    'name': 'stacking_config_test',
    'stacking_list': [
        (config_lightgbm, False),
        (config_lightgbm_new, False),
        (config_xgboost, False),
        (config_xgboost_new, False),
        (config_rf, False),
        (config_extra_tree, False),
        (config_gb, False),
    ],
    'global_force_generate': False,
    'Meta_model': Lightgbm.Lightgbm,
    # predicting parameters
    'model_params': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.1356716009602666,
        'max_bin': 255,
        'metric': 'mae',
        'min_data': 285,
        'min_hessian': 0.11576964737888308,
        'num_boost_round': 500,
        'num_leaves': 10,
        'objective': 'regression',
        'verbose': -1
    },
    # 'resale_offset': 0.012,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, 0),
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': hp.choice('metric', ['mae', 'mse']),
                # 'sub_feature': hp.uniform('sub_feature', 0.1, 0.5),
                'num_leaves': hp.choice('num_leaves', list(range(10, 151, 15))),
                'min_data': hp.choice('min_data', list(range(150, 301, 15))),
                'min_hessian': hp.loguniform('min_hessian', -3, 1),
                'num_boost_round': hp.choice('num_boost_round', [200, 300, 500]),
                'max_bin': hp.choice('max_bin', list(range(50, 151, 10))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1, 0]),
        },
        'max_evals': 1000,
    }
}
