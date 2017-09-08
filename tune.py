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
        'learning_rate': hp.loguniform('learning_rate', -2, -1),
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': hp.choice('metric', ['mae', 'mse']),
        'sub_feature': hp.uniform('sub_feature', 0.03, 0.2),
        'num_leaves': hp.choice('num_leaves', list(range(50, 101, 10))),
        'min_data': hp.choice('min_data', list(range(200, 301, 10))),
        'min_hessian': hp.uniform('min_hessian', 0.3, 0.7),
        'num_boost_round': hp.choice('num_boost_round', [200, 300, 500]),
        'max_bin': hp.choice('max_bin', list(range(100, 301, 50))),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
        'verbose': -1
    },
    'outliers_up_pct': hp.choice('outliers_up_pct', [96, 97, 98, 99, 100]),
    'outliers_lw_pct': hp.choice('outliers_lw_pct', [4, 3, 2, 1, 0])
}

space_xgb = {
    'model_params': {
        'eta': hp.loguniform('eta', -3, 0),
        'gamma': hp.loguniform('gamma', -7, 0),
        'max_depth': hp.choice('max_depth', list(range(1, 7))),
        'min_child_weight': hp.uniform('min_child_weight', 0.1, 5),
        'subsample': hp.choice('subsample', [x/10 for x in range(3, 9)]),
        'colsample_bytree': hp.choice('colsample_bytree', [x/10 for x in range(3, 8)]),
        'colsample_bylevel': hp.choice('colsample_bylevel', [x/10 for x in range(1, 7)]),
        'lambda': hp.choice('lambda', [x/10 for x in range(3, 8)]),
        'alpha': hp.choice('alpha', [x/10 for x in range(0, 5)]),
        'objective': 'reg:linear',
        'eval_metric': hp.choice('eval_metric', ['mae', 'rmse']),
        # 'base_score': y_mean,
        # 'booster': 'gblinear',
        'silent': 1
    },
    'outliers_up_pct': hp.choice('outliers_up_pct', [96, 97, 98, 99, 100]),
    'outliers_lw_pct': hp.choice('outliers_lw_pct', [4, 3, 2, 1, 0])
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
    (XGBoost.XGBoost, configuration['feature_list'], space_xgb, 150, {}),
    # (Lightgbm.Lightgbm, configuration['feature_list'], space_lightgbm, 500, {}),
    # (RFRegressor.RFRegressor, configuration['feature_list'], space_rf, 100, {'clean_na': True}), # 70
]
# feature engineering
for Model, feature_list, parameter_space, max_evals, exp_params in experiments:
    prop = prepare_features(feature_list)
    clean_na = exp_params['clean_na'] if 'clean_na' in exp_params else False
    train_df, X_train_q1_q3, y_train_q1_q3, X_train_q4, y_train_q4 = prepare_training_data(prop, clean_na)
    del train_df; del prop; gc.collect()

    def train_wrapper(params):
        loss = train(X_train_q1_q3, y_train_q1_q3, X_train_q4, y_train_q4,
            Model, **params)
        # return an object to be recorded in hyperopt trails for future uses
        return {
            'loss': loss,
            'status': STATUS_OK,
            'eval_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

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
