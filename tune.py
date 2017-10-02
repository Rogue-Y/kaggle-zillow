# Parameter tuning for single model
# TODO(hzn):
#   *1. better record the tuning process, record each trial
#   *2. better record the useful parameter, features it uses, cv and public lb scores
#   *3. feature selection

from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials

import config
from train import train, prepare_features, prepare_training_data
from ensemble import get_first_layer, stacking

import datetime
import gc
import os
import pickle
import time
from optparse import OptionParser

# parameter space
# lightgbm parameter space
space_lightgbm = {
    'model_params': {
        'learning_rate': hp.loguniform('learning_rate', -2, 0),
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': hp.choice('metric', ['mae', 'mse']),
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
    'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
    'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1])
}

# search space for the so-far best model
# space_xgb = {
#     'model_params': {
#         'eta': hp.loguniform('eta', -3, 0),
#         'gamma': hp.loguniform('gamma', -7, 0),
#         'max_depth': hp.choice('max_depth', list(range(1, 7))),
#         'min_child_weight': hp.uniform('min_child_weight', 0.1, 5),
#         'subsample': hp.choice('subsample', [x/10 for x in range(3, 9)]),
#         'colsample_bytree': hp.choice('colsample_bytree', [x/10 for x in range(3, 8)]),
#         'colsample_bylevel': hp.choice('colsample_bylevel', [x/10 for x in range(1, 7)]),
#         'lambda': hp.choice('lambda', [x/10 for x in range(3, 8)]),
#         'alpha': hp.choice('alpha', [x/10 for x in range(0, 5)]),
#         'objective': 'reg:linear',
#         'eval_metric': hp.choice('eval_metric', ['mae', 'rmse']),
#         # 'base_score': y_mean,
#         # 'booster': 'gblinear',
#         'silent': 1
#     },
#     'outliers_up_pct': hp.choice('outliers_up_pct', [96, 97, 98, 99, 100]),
#     'outliers_lw_pct': hp.choice('outliers_lw_pct', [4, 3, 2, 1, 0])
# }

space_xgb = {
    'model_params': {
        'eta': hp.loguniform('eta', -2, 0),
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
    'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1])
}

space_rf = {
    'model_params': {
        'n_estimators': hp.choice('n_estimators', list(range(10, 31, 10))),
        'criterion': hp.choice('criterion', ['mae', 'mse']),
        'max_features': hp.loguniform('max_features', -2, -1),
        'max_depth': hp.choice('max_depth', list(range(3, 8))),
        'min_samples_split': hp.loguniform('min_samples_split', -4, -2),
        'min_samples_leaf': hp.loguniform('min_samples_leaf', -4, -2),
        # 'n_jobs': -1
    },
    'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
    'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
    'FOLDS': 3 #RF takes long time to train
}

space_ridge = {
    'model_params': {
        'alpha': hp.loguniform('alpha', -2, 2),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
        'random_state': 42
    },
    'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
    'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
    'scaling': True,
    'pca_components': hp.choice('pca_components', [-1, 30, 50, 100, 150, 200]),
}

space_lasso = {
    'model_params': {
        'alpha': hp.loguniform('alpha', -2, 2),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        # 'random_state': 42
    },
    'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
    'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
    'scaling': True,
    'pca_components': hp.choice('pca_components', [-1, 30, 50, 100, 150, 200]),
}

# parameter space for extra tree regressor
space_et = {
    'model_params': {
        'n_estimators': hp.choice('n_estimators', list(range(30, 51, 10))),
        'criterion': hp.choice('criterion', ['mae', 'mse']),
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
}


### Tune models ###

from config import config_linear
# experiments are a list of configs with tunning parameters defined
model_experiments = [
    config_linear
]

def tune_models():
    for config_dict in model_experiments:
        tune_single_model_wrapper(config_dict)

# wrapper of tune_single_model that takes a config dict
def tune_single_model_wrapper(config_dict):
    # Feature list
    feature_list = config_dict['feature_list']
    # Model
    Model = config_dict['Model']
    # clean_na
    clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False
    tune_single_model(Model, feature_list, clean_na, **config_dict['tuning_params'])

def tune_single_model(Model, feature_list, clean_na, parameter_space, max_evals=100, trials=None):
    prop = prepare_features(feature_list, clean_na)
    train_df, transactions = prepare_training_data(prop)
    del transactions; del prop; gc.collect()

    def train_wrapper(params):
        print(params)
        loss = train(train_df, Model, **params)
        # return an object to be recorded in hyperopt trials for future uses
        return {
            'loss': loss,
            'status': STATUS_OK,
            'eval_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'params': params
        }

    if trials is None:
        trials = Trials()
    # tuning parameters
    t1 = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best = fmin(train_wrapper, parameter_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    t2 = time.time()
    print(best)
    print(space_eval(parameter_space, best))
    print("time: %s" %((t2-t1) / 60))

    # save the experiment trials in a pickle
    folder = 'data/trials'
    if not os.path.exists(folder):
        os.makedirs(folder)
    pickle.dump(trials, open("%s/%s_%s_pickle" %(folder, Model.__name__, timestamp), "wb"))

    return trials


### Tune stacking ###

from config import stacking_config_test
stacking_experiments = [
    stacking_config_test
]

def tune_stackings():
    for config_dict in stacking_experiments:
        tune_stacking_wrapper(config_dict)

# wrapper of tune_single_model that takes a config dict
def tune_stacking_wrapper(config_dict):
    # Feature list
    stacking_list = config_dict['stacking_list']
    # Model
    Meta_model = config_dict['Meta_model']
    # Tune
    tune_stacking(stacking_list, Meta_model, **config_dict['tuning_params'])

def tune_stacking(stacking_list, Meta_model, parameter_space, max_evals=100, trials=None):
    first_layer, target, _ = get_first_layer(stacking_list)

    def train_wrapper(params):
        meta_model = Meta_model(model_params=params['model_params'])
        loss = stacking(first_layer, target, meta_model)
        # return an object to be recorded in hyperopt trials for future uses
        return {
            'loss': loss,
            'status': STATUS_OK,
            'eval_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'params': params
        }

    if trials is None:
        trials = Trials()
    # tuning parameters
    t1 = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best = fmin(train_wrapper, parameter_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    t2 = time.time()
    print(best)
    print(space_eval(parameter_space, best))
    print("time: %s" %((t2-t1) / 60))

    # save the experiment trials in a pickle
    folder = 'data/trials/stacking'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open("%s/%s_%s_pickle" %(folder, Meta_model.__name__, timestamp), "wb") as trial_file:
        pickle.dump(trials, trial_file)

    return trials


if __name__ == '__main__':
    # Get configuration
    # parser to parse cmd line option
    parser = OptionParser()
    # tune_model is true by default or when -e flag is present
    parser.add_option('-m', '--model', action='store_true', dest='tune_model', default=True)
    # tune_model is set to false when -s flag is present
    parser.add_option('-t', '--stacking', action='store_false', dest='tune_model')
    # configuration dictionary
    parser.add_option('-c', '--config', action='store', type='string', dest='config_file', default='')
    # parse cmd line arguments
    (options, args) = parser.parse_args()

    config_file = options.config_file
    config_dict = None
    if config_file != '':
        config_dict = getattr(config, config_file)

    if options.tune_model:
        print('Tune models...')
        if config_dict is None:
            # Tune pre-defined experiements if no config is specified on the cmd line
            tune_models()
        else:
            print('Tune config: %s...' %config_dict['name'])
            tune_single_model_wrapper(config_dict)
    else:
        print('Tune stacking...')
        if config_dict is None:
            # Tune pre-defined experiements if no config is specified on the cmd line
            tune_stackings()
        else:
            print('Tune config: %s...' %config_dict['name'])
            tune_stacking_wrapper(config_dict)
