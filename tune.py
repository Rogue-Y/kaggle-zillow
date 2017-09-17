# Parameter tuning for single model
# TODO(hzn):
#   *1. better record the tuning process, record each trial
#   *2. better record the useful parameter, features it uses, cv and public lb scores
#   *3. feature selection

from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials

from config import test_config
from train import train, prepare_features, prepare_training_data
from features import utils, data_clean
from models import XGBoost, Lightgbm, RFRegressor

import datetime
import gc
import os
import pickle
import time

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
        'n_estimators': hp.choice('n_estimators', list(range(10, 151, 30))),
        'criterion': hp.choice('criterion', ['mae', 'mse']),
        'max_features': hp.uniform('max_features', 0.1, 0.6),
        'max_depth': hp.choice('max_depth', [None, *list(range(1, 10, 2))]),
        'min_samples_split': hp.choice('min_samples_split', [3, 10, 30, 100, 300]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [30, 70, 100, 150, 300]),
        'n_jobs': -1
    },
    'outliers_up_pct': hp.choice('outliers_up_pct', [98, 99, 100]),
    'outliers_lw_pct': hp.choice('outliers_lw_pct', [2, 1, 0])
}

# experiments are tuples of format (Model, feature_list, parameter_space, max_run_times, experiment_params)
configuration = test_config
experiments = [
    (XGBoost.XGBoost, configuration['feature_list'], space_xgb, 350, {}),
    # (Lightgbm.Lightgbm, configuration['feature_list'], space_lightgbm, 300, {}),
    # (RFRegressor.RFRegressor, configuration['feature_list'], space_rf, 100, {'clean_na': True}),
]

def tune():
    # feature engineering
    for Model, feature_list, parameter_space, max_evals, exp_params in experiments:
        tune_single_model(Model, feature_list, parameter_space, max_evals, exp_params)

def tune_single_model(Model, feature_list, parameter_space, max_evals, exp_params, trials=None):
    clean_na = exp_params['clean_na'] if 'clean_na' in exp_params else False
    prop = prepare_features(feature_list, clean_na)
    train_df, transactions = prepare_training_data(prop)
    del transactions; del prop; gc.collect()

    def train_wrapper(params):
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

if __name__ == '__main__':
    tune()
