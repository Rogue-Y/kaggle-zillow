from models import Lightgbm, LinearModel
from .config_lightgbm import *
from .config_xgboost import config_xgboost
from .config_ensembles import config_rf_hf_v1, config_gb_hf_v1
from .config_linear import config_linearlasso, config_linearridge, config_elasticnet_hf_v2
from .config_neighbors import config_kneighbors
from .config_catboost import *

from hyperopt import hp

stacking_config_test = {
    'name': 'stacking_config_test',
    'stacking_list': {
        'config': [
            # catboost
            (config_catboost, False),
            (config_catboost_clean, False),
            # linear
            (config_linearridge, False),
            (config_linearlasso, False),
            (config_elasticnet_hf_v2, False),
            # lightgbm
            (config_lightgbm_all_regression_l2_dart, False),
            (config_lightgbm_all_regression_l1_dart, False),
            (config_lightgbm_all_huber, False),
            (config_lightgbm_all_fair_dart, False),
            # xgboost
            (config_xgboost, False),
            # knn
            (config_kneighbors, False),
            (config_rf_hf_v1, False),
            # (config_extra_tree, False),
            (config_gb_hf_v1, False),
        ],
        'csv': [

        ]
    },
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
                'objective': hp.choice('objective', ['regression_l2', 'regression_l1', 'huber', 'fair']),
                # 'metric': hp.choice('metric', ['mae', 'mse']),
                # 'sub_feature': hp.choice('sub_feature', [0.5, 0.6, 0.7, 0.8, 0.9, 1]),
                'num_leaves': hp.choice('num_leaves', [5, 15, 45, 90, 135]),
                'min_data': hp.choice('min_data', [30, 100, 300, 900]),
                'min_hessian': hp.loguniform('min_hessian', -1, 1),
                'num_boost_round': hp.choice('num_boost_round', [300, 500, 1000]),
                'max_bin': hp.choice('max_bin', [100, 300, 500, 750]),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1,
                'categorical_feature': []
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [94, 95, 96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [6, 5, 4, 3, 2, 1, 0]),
        },
        'max_evals': 100,
    }
}


stacking_config_linear = {
    'name': 'stacking_config_linear',
    'stacking_list': {
        'config': [
            # catboost
            (config_catboost, False),
            (config_catboost_clean, False),
            # linear
            (config_linearridge, False),
            (config_linearlasso, False),
            (config_elasticnet_hf_v2, False),
            # lightgbm
            (config_lightgbm_all_regression_l2_dart, False),
            (config_lightgbm_all_regression_l1_dart, False),
            (config_lightgbm_all_huber, False),
            (config_lightgbm_all_fair_dart, False),
            # xgboost
            (config_xgboost, False),
            # knn
            (config_kneighbors, False),
            (config_rf_hf_v1, False),
            # (config_extra_tree, False),
            (config_gb_hf_v1, False),
        ],
        'csv': [

        ]
    },
    'global_force_generate': False,
    'Meta_model': LinearModel.Linear,
    'clean_na': True,
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
        'max_evals': 30
    }
}

# 0.0637986604067, 0.0685674588323, 0.0669778593571
stacking_config_ridge = {
    'name': 'stacking_config_ridge',
    'stacking_list': {
        'config': [
            # catboost
            (config_catboost, False),
            (config_catboost_clean, False),
            # linear
            (config_linearridge, False),
            (config_linearlasso, False),
            (config_elasticnet_hf_v2, False),
            # lightgbm
            (config_lightgbm_all_regression_l2_dart, False),
            (config_lightgbm_all_regression_l1_dart, False),
            (config_lightgbm_all_huber, False),
            (config_lightgbm_all_fair_dart, False),
            # xgboost
            (config_xgboost, False),
            # knn
            (config_kneighbors, False),
            (config_rf_hf_v1, False),
            # (config_extra_tree, False),
            (config_gb_hf_v1, False),
        ],
        'csv': [

        ]
    },
    'global_force_generate': False,
    'clean_na': True,
    'Meta_model': LinearModel.RidgeRegressor,
    # predicting parameters
    'model_params': {
        'alpha': 0.35479842518622096,
        'fit_intercept': True,
        'random_state': 42,
        'solver': 'sparse_cg',
        'tol': 0.00711606311467554
    },
    'outliers_lw_pct': 2,
    'outliers_up_pct': 97,
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
        'max_evals': 3000
    }
}

all_configs = [
    # catboost
    (config_catboost, False),
    (config_catboost_clean, False),
    # linear
    (config_linearridge, False),
    (config_linearlasso, False),
    (config_elasticnet_hf_v2, False),
    # lightgbm
    (config_lightgbm_all_regression_l2_dart, False),
    (config_lightgbm_all_regression_l1_dart, False),
    (config_lightgbm_all_huber, False),
    (config_lightgbm_all_fair_dart, False),
    # xgboost
    (config_xgboost, False),
    # knn
    (config_kneighbors, False),
    (config_rf_hf_v1, False),
    # (config_extra_tree, False),
    (config_gb_hf_v1, False),
]

from features import stacking_feature_list

# 0.0637807268378, 0.0684604519817, 0.0669005436004
stacking_config_ridge_with_feature = {
    'name': 'stacking_config_ridge_with_feature',
    'stacking_list': {
        'config': all_configs,
        'csv': []
    },
    'global_force_generate': False,
    'feature_list': stacking_feature_list.feature_list_ridge1,
    'clean_na': True,
    'Meta_model': LinearModel.RidgeRegressor,
    # predicting parameters
    # 'resale_offset': 0.012,
    'model_params': {
        'alpha': 0.21679100712172433,
        'fit_intercept': True,
        'normalize': True,
        'random_state': 42,
        'solver': 'sparse_cg',
        'tol': 0.006812842662686247
    },
    'outliers_lw_pct': 3,
    'outliers_up_pct': 96,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -3, 2),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
                'tol': hp.loguniform('tol', -6, -2),
                'normalize': hp.choice('normalize', [True, False]),
                'random_state': 42
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [94, 95, 96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [6, 5, 4, 3, 2, 1, 0]),
        },
        'max_evals': 1000
    }
}

# 0.0637944738348, 0.0684538241452, 0.06690070737508784
stacking_config_ridge_with_feature3 = {
    'name': 'stacking_config_ridge_with_feature3',
    'stacking_list': {
        'config': all_configs,
        'csv': []
    },
    'global_force_generate': False,
    'feature_list': stacking_feature_list.feature_list_ridge3,
    'clean_na': True,
    'Meta_model': LinearModel.RidgeRegressor,
    # predicting parameters
    # 'resale_offset': 0.012,
    'model_params': {
        'alpha': 0.26575111538470936,
        'fit_intercept': True,
        'normalize': True,
        'random_state': 42,
        'solver': 'sag',
        'tol': 0.08947065085372981,
        'max_iter': 10000,
    },
    'outliers_lw_pct': 3,
    'outliers_up_pct': 96,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -3, 2),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
                'tol': hp.loguniform('tol', -6, -2),
                'normalize': hp.choice('normalize', [True, False]),
                'random_state': 42,
                'max_iter': 5000,
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1, 0]),
        },
        'max_evals': 2000
    }
}


partial_configs = [
    # catboost
    (config_catboost, False),
    (config_catboost_clean, False),
    # linear
    # (config_linearridge, False),
    # (config_linearlasso, False),
    # (config_elasticnet_hf_v2, False),
    # lightgbm
    (config_lightgbm_all_regression_l2_dart, False),
    (config_lightgbm_all_regression_l1_dart, False),
    (config_lightgbm_all_huber, False),
    (config_lightgbm_all_fair_dart, False),
    # xgboost
    (config_xgboost, False),
    # knn
    # (config_kneighbors, False),
    (config_rf_hf_v1, False),
    # (config_extra_tree, False),
    (config_gb_hf_v1, False),
]

# 0.0637610460772, 0.0684508548897: 0.06688758528552376
stacking_config_ridge_partial_config_with_feature = {
    'name': 'stacking_config_ridge_partial_config_with_feature',
    'stacking_list': {
        'config': partial_configs + [(config_catboost_clean_long, False)],
        'csv': []
    },
    'global_force_generate': False,
    'feature_list': stacking_feature_list.feature_list_ridge1,
    'clean_na': True,
    'Meta_model': LinearModel.RidgeRegressor,
    # predicting parameters
    # 'resale_offset': 0.012,
    'model_params': {
        'alpha': 0.1576368858337792,
        'fit_intercept': True,
        'normalize': True,
        'random_state': 42,
        'solver': 'sag',
        'tol': 0.10128973680416681
    },
    'outliers_lw_pct': 3,
    'outliers_up_pct': 96,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -3, 2),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
                'tol': hp.loguniform('tol', -6, -2),
                'normalize': hp.choice('normalize', [True, False]),
                'random_state': 42
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [94, 95, 96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [6, 5, 4, 3, 2, 1, 0]),
        },
        'max_evals': 3000
    }
}

# 0.0637408563565, 0.0684353548124: 0.06687052199374166
stacking_config_ridge_partial_config_with_feature_long = {
    'name': 'stacking_config_ridge_partial_config_with_feature_long',
    'stacking_list': {
        'config': partial_configs + [(config_catboost_clean_long, False)],
        'csv': []
    },
    'global_force_generate': False,
    'feature_list': stacking_feature_list.feature_list_ridge1,
    'clean_na': True,
    'Meta_model': LinearModel.RidgeRegressor,
    # predicting parameters
    # 'resale_offset': 0.012,
    'model_params': {
        'alpha': 0.1603569776463936,
        'fit_intercept': True,
        'normalize': True,
        'random_state': 42,
        'solver': 'sag',
        'tol': 0.11676574578298685
    },
    'outliers_lw_pct': 3,
    'outliers_up_pct': 96,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -2, -1),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
                'tol': hp.loguniform('tol', -4, -1),
                'normalize': True,
                'random_state': 42
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [4, 3, 2, 1, 0]),
        },
        'max_evals': 3000
    }
}

# 0.0637929003149, 0.0684426681328, 0.06689274552684045
stacking_config_ridge_partial_config_with_feature2 = {
    'name': 'stacking_config_ridge_partial_config_with_feature2',
    'stacking_list': {
        'config': partial_configs,
        'csv': []
    },
    'global_force_generate': False,
    'feature_list': stacking_feature_list.feature_list_ridge3,
    'clean_na': True,
    'Meta_model': LinearModel.RidgeRegressor,
    # predicting parameters
    # 'resale_offset': 0.012,
    'model_params': {
        'alpha': 0.138140369883881,
        'fit_intercept': True,
        'normalize': True,
        'random_state': 42,
        'solver': 'sparse_cg',
        'tol': 0.012776912325107864
    },
    'outliers_lw_pct': 3,
    'outliers_up_pct': 96,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -3, 2),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
                'tol': hp.loguniform('tol', -6, -2),
                'normalize': hp.choice('normalize', [True, False]),
                'random_state': 42
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [94, 95, 96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [6, 5, 4, 3, 2, 1, 0]),
        },
        'max_evals': 2500
    }
}

# 0.0637659790338, 0.0684294079447: 0.06687493164106624
stacking_config_ridge_partial_config_with_feature_long2 = {
    'name': 'stacking_config_ridge_partial_config_with_feature_long2',
    'stacking_list': {
        'config': partial_configs + [(config_catboost_clean_long, False)],
        'csv': []
    },
    'global_force_generate': False,
    'feature_list': stacking_feature_list.feature_list_ridge3,
    'clean_na': True,
    'Meta_model': LinearModel.RidgeRegressor,
    # predicting parameters
    # 'resale_offset': 0.012,
    'model_params': {
        'alpha': 0.18802452689879962,
        'fit_intercept': True,
        'normalize': True,
        'random_state': 42,
        'solver': 'sag',
        'tol': 0.12728836824138723
    },
    'outliers_lw_pct': 3,
    'outliers_up_pct': 96,

    # tuning parameters
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -2, -1),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
                'tol': hp.loguniform('tol', -4, -1),
                'normalize': True,
                'random_state': 42
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [96, 97, 98, 99, 100]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [4, 3, 2, 1, 0]),
        },
        'max_evals': 2000
    }
}

# # 0.063781907088, 0.0685477473751, 0.06695913394605855
# stacking_config_ridge_partial_config = {
#     'name': 'stacking_config_ridge_partial_config',
#     'stacking_list': {
#         'config': partial_configs,
#         'csv': []
#     },
#     'global_force_generate': False,
#     'clean_na': True,
#     'Meta_model': LinearModel.RidgeRegressor,
#     # predicting parameters
#     # 'resale_offset': 0.012,
#
#     # tuning parameters
#     'tuning_params': {
#         'parameter_space': {
#             'model_params': {
#                 'alpha': hp.loguniform('alpha', -2, -1),
#                 'fit_intercept': hp.choice('fit_intercept', [True, False]),
#                 'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
#                 'tol': hp.loguniform('tol', -4, -1),
#                 'normalize': True,
#                 'random_state': 42
#             },
#             'outliers_up_pct': hp.choice('outliers_up_pct', [96, 97, 98, 99, 100]),
#             'outliers_lw_pct': hp.choice('outliers_lw_pct', [4, 3, 2, 1, 0]),
#         },
#         'max_evals': 3000
#     }
# }

# 'model_params': {'boosting_type': 'gbdt', 'categorical_feature': (), 'learning_rate': 0.007271947435414975, 'max_bin': 210, 'metric': 'mse', 'min_data': 270, 'min_hessian': 1.5770551255125693, 'num_boost_round': 500, 'num_leaves': 5, 'objective': 'regression_l1', 'sub_feature': 0.5, 'verbose': -1}, 'outliers_lw_pct': 2, 'outliers_up_pct': 98,
