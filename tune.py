# Parameter tuning for single model
# TODO(hzn):
#   *1. better record the tuning process, record each trial
#   *2. better record the useful parameter, features it uses, cv and public lb scores
#   *3. feature selection

from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials

import config
from train import train_process, get_dfs
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

from config import config_linearlasso
# experiments are a list of configs with tunning parameters defined
model_experiments = [
    config_linearlasso
]

def tune_models():
    for config_dict in model_experiments:
        tune_single_model_wrapper(config_dict)

# wrapper of tune_single_model that takes a config dict
# def tune_single_model_wrapper(config_dict, trials=None):
#     config_name = config_dict['name']
#     # Feature list
#     feature_list = config_dict['feature_list']
#     # Model
#     Model = config_dict['Model']
#     # clean_na
#     clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False
#     tune_single_model(Model, feature_list, clean_na, config_name, **config_dict['tuning_params'], trials=trials)

def tune_single_model(config_dict, trials=None):
    df2016, df_all, _, _ = get_dfs(config_dict)
    Model = config_dict['Model']
    config_name = config_dict['name']
    parameter_space = config_dict['tuning_params']['parameter_space']
    max_evals = config_dict['tuning_params']['max_evals']

    def train_wrapper(params):
        print(params)
        loss = train_process(df2016, df_all, Model, params, 'tune')
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
    print('best trial get at round: ' + str(trials.best_trial['tid']))
    print('best loss: ' + str(trials.best_trial['result']['loss']))
    print(best)
    print(space_eval(parameter_space, best))
    print("time: %s" %((t2-t1) / 60))

    # save the experiment trials in a pickle
    folder = 'data/trials'
    if not os.path.exists(folder):
        os.makedirs(folder)
    pickle.dump(trials, open("%s/%s_%s_pickle" %(folder, config_name, timestamp), "wb"))

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
def tune_stacking_wrapper(config_dict, trials=None):
    config_name = config_dict['name']
    # Feature list
    stacking_list = config_dict['stacking_list']
    # Model
    Meta_model = config_dict['Meta_model']
    # whether force generate all first layer
    force_generate = config_dict['global_force_generate'] if 'global_force_generate' in config_dict else False
    # Tune
    tune_stacking(stacking_list, Meta_model, force_generate, config_name, **config_dict['tuning_params'], trials=trials)

def tune_stacking(stacking_list, Meta_model, force_generate, config_name, parameter_space, max_evals=100, trials=None):
    first_layer, target, _ = get_first_layer(stacking_list, global_force_generate=force_generate)

    def train_wrapper(params):
        meta_model = Meta_model(model_params=params['model_params'])
        outliers_up_pct = params['outliers_up_pct'] if 'outliers_up_pct' in params else 100
        outliers_lw_pct = params['outliers_lw_pct'] if 'outliers_lw_pct' in params else 0
        loss = stacking(first_layer, target, meta_model, outliers_lw_pct, outliers_up_pct)
        # return an object to be recorded in hyperopt trials for future uses
        return {
            'loss': loss,
            'status': STATUS_OK,
            'eval_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'params': params
        }

    if trials is None:
        trials = Trials()
    print(len(trials.trials))
    # tuning parameters
    t1 = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best = fmin(train_wrapper, parameter_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    t2 = time.time()
    print('best trial get at round: ' + str(trials.best_trial['tid']))
    print('best loss: ' + str(trials.best_trial['result']['loss']))
    print(best)
    print(space_eval(parameter_space, best))
    print("time: %s" %((t2-t1) / 60))

    # save the experiment trials in a pickle
    folder = 'data/trials/stacking'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open("%s/%s_%s_pickle" %(folder, config_name, timestamp), "wb") as trial_file:
        pickle.dump(trials, trial_file)

    return trials


if __name__ == '__main__':
    # Get configuration
    # parser to parse cmd line option
    parser = OptionParser()
    # tune_model is true by default or when -e flag is present
    parser.add_option('-m', '--model', action='store_true', dest='tune_model', default=True)
    # tune_model is set to false when -s flag is present
    parser.add_option('-s', '--stacking', action='store_false', dest='tune_model')
    # configuration dictionary
    parser.add_option('-c', '--config', action='store', type='string', dest='config_file', default='')
    # trials of the existing tuning, only used when tuning single model config or single stacking config
    parser.add_option('-t', '--trials', action='store', type='string', dest='trials_file', default='')
    # parse cmd line arguments
    (options, args) = parser.parse_args()

    config_file = options.config_file
    config_dict = None
    if config_file != '':
        config_dict = getattr(config, config_file)

    trials_file = options.trials_file
    trials = None
    if trials_file != '':
        trials_folder = 'data/trials' if options.tune_model else 'data/trials/stacking'
        trials_path = '%s/%s' %(trials_folder, trials_file)
        if os.path.exists(trials_path):
            print('Using trials: %s' %trials_path)
            trials = pickle.load(open(trials_path, 'rb'))

    if options.tune_model:
        print('Tune models...')
        if config_dict is None:
            # Tune pre-defined experiements if no config is specified on the cmd line
            tune_models()
        else:
            print('Tune config: %s...' %config_dict['name'])
            tune_single_model(config_dict, trials)
    else:
        print('Tune stacking...')
        if config_dict is None:
            # Tune pre-defined experiements if no config is specified on the cmd line
            tune_stackings()
        else:
            print('Tune config: %s...' %config_dict['name'])
            tune_stacking_wrapper(config_dict, trials)
