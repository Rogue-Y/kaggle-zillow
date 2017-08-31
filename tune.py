from hyperopt import hp, fmin, tpe, space_eval

from config import test_config
from train import train, prepare_features

# feature engineering
configuration = test_config
prop = prepare_features(configuration['feature_list'])

# parameter space
space = {
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

def train_wrapper(params):
    return train(prop, configuration['model'], **params)

# tuning parameters
best = fmin(train_wrapper, space, algo=tpe.suggest, max_evals=300)

print(best)
print(space_eval(space, best))
