from hyperopt import hp, fmin, tpe, space_eval

from config import test_config
from train import train_for_tuning, prepare_features
from features import utils, data_clean

import gc

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
    }
}

space_xgb = {
    'model_params': {
        'eta': hp.loguniform('eta', -3, 0),
        'gamma': hp.uniform('gamma', 0, 2),
        'max_depth': hp.choice('max_depth', list(range(1, 6))),
        'min_child_weight': hp.uniform('min_child_weight', 0.1, 5),
        'subsample': hp.choice('subsample', [x/10 for x in range(3, 8)]),
        'colsample_bytree': hp.choice('colsample_bytree', [x/10 for x in range(3, 8)]),
        'colsample_bylevel': hp.choice('colsample_bylevel', [x/10 for x in range(3, 8)]),
        'lambda': hp.choice('lambda', [x/10 for x in range(3, 8)]),
        'alpha': hp.choice('alpha', [x/10 for x in range(3, 8)]),
        'objective': 'reg:linear',
        'eval_metric': hp.choice('eval_metric', ['mae', 'rmse']),
        # 'base_score': y_mean,
        # 'booster': 'gblinear',
        'silent': 1
    },
    'outliers_up_pct': hp.choice('outliers_up_pct', [98, 99, 100]),
    'outliers_lw_pct': hp.choice('outliers_lw_pct', [2, 1, 0])
}

# feature engineering
configuration = test_config
prop = prepare_features(configuration['feature_list'])
train_df, X_train_q1_q3, y_train_q1_q3, X_train_q4, y_train_q4 = prepare_training_data(prop)
del train_df; del prop; gc.collect()

def train_wrapper(params):
    return train(X_train_q1_q3, y_train_q1_q3, X_train_q4, y_train_q4,
        configuration['model'], **params)

# tuning parameters
best = fmin(train_wrapper, space_xgb, algo=tpe.suggest, max_evals=200)

print(best)
print(space_eval(space_xgb, best))
