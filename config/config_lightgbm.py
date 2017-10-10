# lightgbm config
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_non_linear
# model
from models import Lightgbm
# for tunning parameters
from hyperopt import hp


# Configuration:
config_lightgbm = {
    'name': 'config_lightgbm',
    # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
    # 'resale_offset': 0,
    'feature_list': feature_list_non_linear.feature_list,
    'Model': Lightgbm.Lightgbm,
    # Best lightgbm param with geo_neighborhood and geo_zip, 0.0646722050526
    # 'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.012923556870735842,
    #     'metric': 'mse', 'min_data': 200, 'min_hessian': 0.232705809294419,
    #     'num_boost_round': 500, 'num_leaves': 30, 'objective': 'regression',
    #     'sub_feature': 0.1596488603529622, 'verbose': -1},
    # Best lightgbm param with almost all features, 0.0646671968399
    # 'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.05525896243523381,
    #     'metric': 'mae', 'min_data': 260, 'min_hessian': 0.57579034653711,
    #     'num_boost_round': 300, 'num_leaves': 70, 'objective': 'regression_l1',
    #     'sub_feature': 0.06638755200543586, 'verbose': -1},
    'training_params': {
        'FOLDS': 2,
        'model_params': {
            'boosting_type': 'gbdt',
            'learning_rate': 0.1392149300094899,
            'max_bin': 130,
            'metric': 'mae',
            'min_data': 255,
            'min_hessian': 0.2372321993762161,
            'num_boost_round': 300,
            'num_leaves': 10,
            'objective': 'regression',
            'sub_feature': 0.1228828936613017,
            'verbose': -1
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 96
    },
    'stacking_params': {
        'FOLDS': 2,
        'model_params': {
            'boosting_type': 'gbdt',
            'learning_rate': 0.1392149300094899,
            'max_bin': 130,
            'metric': 'mae',
            'min_data': 255,
            'min_hessian': 0.2372321993762161,
            'num_boost_round': 300,
            'num_leaves': 10,
            'objective': 'regression',
            'sub_feature': 0.1228828936613017,
            'verbose': -1
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 96
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, 0),
                'boosting_type': 'gbdt',
                'objective': 'regression',
                # 'metric': hp.choice('metric', ['mae', 'mse']),
                'sub_feature': hp.uniform('sub_feature', 0.1, 0.5),
                'num_leaves': hp.choice('num_leaves', list(range(10, 151, 15))),
                'min_data': hp.choice('min_data', list(range(150, 301, 15))),
                'min_hessian': hp.loguniform('min_hessian', -3, 1),
                'num_boost_round': hp.choice('num_boost_round', [200, 300, 500]),
                'max_bin': hp.choice('max_bin', list(range(50, 151, 10))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1
            },
            'FOLDS': 3,
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1])
        },
        'max_evals': 325
    }
}

feature_list_all_categorical_feature =  [
    'airconditioningtypeid',
    'architecturalstyletypeid',
    'buildingclasstypeid',
    'decktypeid',
    'fips',
    'fireplaceflag',
    'hashottuborspa',
    'heatingorsystemtypeid',
    'poolcnt',
    'pooltypeid10',
    'pooltypeid2',
    'pooltypeid7',
    'propertycountylandusecode',
    'propertylandusetypeid',
    'propertyzoningdesc',
    'regionidcounty',
    'regionidcity',
    'regionidzip',
    'regionidneighborhood',
    'storytypeid',
    'typeconstructiontypeid',
    'taxdelinquencyflag',
    'boolean_has_ac',
    'boolean_has_garage_pool_and_ac',
    'boolean_has_heat',
    'built_before_year',
    'has_fireplace',
    'is_garagetotalsqft_zero',
    'has_partial_garagecarcnt',
    'is_unitcnt_gt_four',
    'has_shed_in_yard',
    'is_numberofstories_gt_three',
    'is_assessmentyear_2015',
    'is_tax_assessed',
    'is_taxdelinquencyyear_before_2014',
    'has_construction_type',
    'is_roomcnt_zero'
]

# Configuration:
config_lightgbm_all_regression_l2 = {
    'name': 'config_lightgbm_all_regression_l2',
    'feature_list': feature_list_non_linear.feature_list_all,
    'Model': Lightgbm.Lightgbm,
    'training_params': {
        'model_params': {
            'boosting_type': 'gbdt',
            'learning_rate': 0.13560810846153493,
            'max_bin': 90,
            'min_data': 300,
            'min_hessian': 0.3329659425303045,
            'num_boost_round': 300,
            'num_leaves': 10,
            'objective': 'regression_l2',
            'sub_feature': 0.2467205146666428,
            'verbose': -1
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 96
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, -1),
                'boosting_type': 'gbdt',
                'objective': 'regression_l2',
                # 'metric': hp.choice('metric', ['mae', 'mse']),
                'sub_feature': hp.uniform('sub_feature', 0.1, 0.5),
                'num_leaves': hp.choice('num_leaves', list(range(10, 151, 15))),
                'min_data': hp.choice('min_data', list(range(150, 301, 15))),
                # 'min_hessian': hp.loguniform('min_hessian', -2, -1),
                'num_boost_round': hp.choice('num_boost_round', [350, 500, 700]),
                'max_bin': hp.choice('max_bin', list(range(100, 351, 50))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1,
                # categorical features
                'categorical_feature': feature_list_all_categorical_feature,
                'max_cat_group': hp.choice('max_cat_group', [32, 64]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1])
        },
        'max_evals': 150
    }
}

# Configuration:
config_lightgbm_all_regression_l1 = {
    'name': 'config_lightgbm_all_regression_l1',
    'feature_list': feature_list_non_linear.feature_list_all,
    'Model': Lightgbm.Lightgbm,
    'training_params': {
        'model_params': {
            'boosting_type': 'gbdt',
            'learning_rate': 0.13566555825466994,
            'max_bin': 50,
            'min_data': 225,
            'min_hessian': 0.09169556339297391,
            'num_boost_round': 200,
            'num_leaves': 25,
            'objective': 'regression_l1',
            'sub_feature': 0.14455184302330204,
            'verbose': -1
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 97
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, -1),
                'boosting_type': 'gbdt',
                'objective': 'regression_l1',
                # l1 parameters
                'gaussian_eta': hp.loguniform('gaussian_eta', -1, 0),
                # 'metric': hp.choice('metric', ['mae', 'mse']),
                'sub_feature': hp.uniform('sub_feature', 0.1, 0.5),
                'num_leaves': hp.choice('num_leaves', list(range(10, 151, 15))),
                'min_data': hp.choice('min_data', list(range(150, 301, 15))),
                # 'min_hessian': hp.loguniform('min_hessian', -2, -1),
                'num_boost_round': hp.choice('num_boost_round', [350, 500, 700]),
                'max_bin': hp.choice('max_bin', list(range(100, 351, 50))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1,
                # categorical features
                'categorical_feature': feature_list_all_categorical_feature,
                'max_cat_group': hp.choice('max_cat_group', [32, 64]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1])
        },
        'max_evals': 300
    }
}

# Configuration:
config_lightgbm_all_regression_l1_dart = {
    'name': 'config_lightgbm_all_regression_l1_dart',
    'feature_list': feature_list_non_linear.feature_list_all,
    'Model': Lightgbm.Lightgbm,
    'training_params': {
        'model_params': {
            'boosting_type': 'dart',
            'learning_rate': 0.13566555825466994,
            'max_bin': 50,
            'min_data': 225,
            'min_hessian': 0.09169556339297391,
            'num_boost_round': 200,
            'num_leaves': 25,
            'objective': 'regression_l1',
            'sub_feature': 0.14455184302330204,
            'verbose': -1,
            'drop_rate': 0.2,
            'skip_drop': 0.6,
            'max_drop': 60,
            'categorical_feature': feature_list_all_categorical_feature,
            'max_cat_group': 32,
        },
        'outliers_lw_pct': 5,
        'outliers_up_pct': 97
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, -1),
                'boosting_type': 'dart',
                'objective': 'regression_l1',
                # l1 parameters
                'gaussian_eta': hp.loguniform('gaussian_eta', -1, 0),
                # 'metric': hp.choice('metric', ['mae', 'mse']),
                'sub_feature': hp.uniform('sub_feature', 0.1, 0.5),
                'num_leaves': hp.choice('num_leaves', list(range(10, 151, 15))),
                'min_data': hp.choice('min_data', list(range(150, 301, 15))),
                # 'min_hessian': hp.loguniform('min_hessian', -2, -1),
                'num_boost_round': hp.choice('num_boost_round', [350, 500, 700]),
                'max_bin': hp.choice('max_bin', list(range(100, 351, 50))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1,
                # dart parameters
                'drop_rate': hp.choice('drop_rate', [0.1, 0.2]),
                'skip_drop': hp.choice('skip_drop', [0.3, 0.5, 0.7]),
                # 'max_drop': 60,
                # categorical features
                'categorical_feature': feature_list_all_categorical_feature,
                'max_cat_group': hp.choice('max_cat_group', [32, 64]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1])
        },
        'max_evals': 300
    }
}

# Configuration:
config_lightgbm_all_huber = {
    'name': 'config_lightgbm_all_huber',
    'feature_list': feature_list_non_linear.feature_list_all,
    'Model': Lightgbm.Lightgbm,
    'training_params': {
    },
    'stacking_params': {
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, -1),
                'boosting_type': 'gbdt',
                'objective': 'huber',
                # huber parameters
                'huber_delta': hp.uniform('huber_delta', 0.2, 1),
                'gaussian_eta': hp.loguniform('gaussian_eta', -1, 0),
                # 'metric': hp.choice('metric', ['mae', 'mse']),
                'sub_feature': hp.uniform('sub_feature', 0.1, 0.5),
                'num_leaves': hp.choice('num_leaves', list(range(10, 151, 15))),
                'min_data': hp.choice('min_data', list(range(150, 301, 15))),
                # 'min_hessian': hp.loguniform('min_hessian', -2, -1),
                'num_boost_round': hp.choice('num_boost_round', [350, 500, 700]),
                'max_bin': hp.choice('max_bin', list(range(100, 351, 50))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1,
                # categorical features
                'categorical_feature': feature_list_all_categorical_feature,
                'max_cat_group': hp.choice('max_cat_group', [32, 64]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1])
        },
        'max_evals': 150
    }
}

# Configuration:
config_lightgbm_all_fair = {
    'name': 'config_lightgbm_all_fair',
    'feature_list': feature_list_non_linear.feature_list_all,
    'Model': Lightgbm.Lightgbm,
    'training_params': {
    },
    'stacking_params': {
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, -1),
                'boosting_type': 'gbdt',
                'objective': 'fair',
                # fair parameters
                'fair_c': hp.uniform('fair_c', 0.5, 2),
                # huber parameters
                'huber_delta': hp.uniform('huber_delta', 0.2, 1),
                'gaussian_eta': hp.loguniform('gaussian_eta', -1, 0),
                # 'metric': hp.choice('metric', ['mae', 'mse']),
                'sub_feature': hp.uniform('sub_feature', 0.1, 0.5),
                'num_leaves': hp.choice('num_leaves', list(range(10, 151, 15))),
                'min_data': hp.choice('min_data', list(range(150, 301, 15))),
                # 'min_hessian': hp.loguniform('min_hessian', -2, -1),
                'num_boost_round': hp.choice('num_boost_round', [350, 500, 700]),
                'max_bin': hp.choice('max_bin', list(range(100, 351, 50))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1,
                # categorical features
                'categorical_feature': feature_list_all_categorical_feature,
                'max_cat_group': hp.choice('max_cat_group', [32, 64]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1])
        },
        'max_evals': 150
    }
}

# # Configuration:
# config_lightgbm_all_poisson = {
#     'name': 'config_lightgbm_all_poisson',
#     'feature_list': feature_list_non_linear.feature_list_all,
#     'Model': Lightgbm.Lightgbm,
#     'training_params': {
#     },
#     'stacking_params': {
#     },
#     'tuning_params': {
#         'parameter_space': {
#             'model_params': {
#                 'learning_rate': hp.loguniform('learning_rate', -2, 0),
#                 'boosting_type': 'gbdt',
#                 'objective': 'poisson',
#                 # 'metric': hp.choice('metric', ['mae', 'mse']),
#                 'sub_feature': hp.uniform('sub_feature', 0.1, 0.5),
#                 'num_leaves': hp.choice('num_leaves', list(range(10, 151, 15))),
#                 'min_data': hp.choice('min_data', list(range(150, 301, 15))),
#                 'min_hessian': hp.loguniform('min_hessian', -3, 1),
#                 'num_boost_round': hp.choice('num_boost_round', [200, 300, 500]),
#                 'max_bin': hp.choice('max_bin', list(range(50, 151, 10))),
#                 # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
#                 # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
#                 'verbose': -1
#             },
#             'FOLDS': 3,
#             'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
#             'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1])
#         },
#         'max_evals': 150
#     }
# }


from features import feature_list_non_linear_geo
# Configuration:
config_lightgbm_geo = {
    'name': 'config_lightgbm_geo',
    'feature_list': feature_list_non_linear_geo.feature_list,
    'Model': Lightgbm.Lightgbm,
    'training_params': {
        'FOLDS': 3,
        'model_params': {
            'boosting_type': 'gbdt',
            'learning_rate': 0.14570480513583217,
            'max_bin': 100,
            'metric': 'mse',
            'min_data': 240,
            'min_hessian': 0.4281167522186269,
            'num_boost_round': 300,
            'num_leaves': 10,
            'objective': 'regression',
            'sub_feature': 0.16696129694986633,
            'verbose': -1
        },
        'outliers_lw_pct': 6,
        'outliers_up_pct': 97
    },
    'stacking_params': {
        'FOLDS': 3,
        'model_params': {
            'boosting_type': 'gbdt',
            'learning_rate': 0.14570480513583217,
            'max_bin': 100,
            'metric': 'mse',
            'min_data': 240,
            'min_hessian': 0.4281167522186269,
            'num_boost_round': 300,
            'num_leaves': 10,
            'objective': 'regression',
            'sub_feature': 0.16696129694986633,
            'verbose': -1
        },
        'outliers_lw_pct': 6,
        'outliers_up_pct': 97
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, 0),
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': hp.choice('metric', ['mae', 'mse']),
                'sub_feature': hp.loguniform('sub_feature', -2, -1),
                'num_leaves': hp.choice('num_leaves', list(range(10, 51, 10))),
                'min_data': hp.choice('min_data', list(range(200, 301, 10))),
                'min_hessian': hp.loguniform('min_hessian', -2, 0),
                'num_boost_round': hp.choice('num_boost_round', [200, 300, 500]),
                'max_bin': hp.choice('max_bin', list(range(100, 201, 10))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1
            },
            'FOLDS': 3,
            'outliers_up_pct': hp.choice('outliers_up_pct', [94, 95, 96, 97, 98]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [6, 5, 4, 3, 2])
        },
        'max_evals': 60
    }
}

# Configuration:
config_lightgbm_new = {
    'name': 'config_lightgbm_new',
    'feature_list': feature_list_non_linear.feature_list,
    'Model': Lightgbm.Lightgbm,
    'training_params': {
        'FOLDS': 3,
        'model_params': {
            'boosting_type': 'gbdt',
            'learning_rate': 0.22198085859401054,
            'max_bin': 60,
            'metric': 'mse',
            'min_data': 210,
            'min_hessian': 0.7016584408191289,
            'num_boost_round': 200,
            'num_leaves': 10,
            'objective': 'regression',
            'sub_feature': 0.13346631904716155,
            'verbose': -1
        },
        'outliers_lw_pct': 2,
        'outliers_up_pct': 98
    },
    'stacking_params': {
        'FOLDS': 3,
        'model_params': {
            'boosting_type': 'gbdt',
            'learning_rate': 0.22198085859401054,
            'max_bin': 60,
            'metric': 'mse',
            'min_data': 210,
            'min_hessian': 0.7016584408191289,
            'num_boost_round': 200,
            'num_leaves': 10,
            'objective': 'regression',
            'sub_feature': 0.13346631904716155,
            'verbose': -1
        },
        'outliers_lw_pct': 2,
        'outliers_up_pct': 98
    }
}
