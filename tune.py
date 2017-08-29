from hyperopt import hp, fmin, tpe, space_eval

from config import test_config
from train import train_wrapper

configuration = test_config

space = {
    'Model': configuration['model'],
    'model_params': {
        'learning_rate': hp.loguniform('learning_rate', -3, -1),
        'boosting_type': 'gbdt',
        'objective': hp.choice('objective', ['regression_l2', 'regression_l1']),
        'metric': hp.choice('metric', ['mae', 'mse']),
        'sub_feature': hp.uniform('sub_feature', 0.05, 0.5),
        'num_leaves': hp.choice('num_leaves', list(range(10, 101, 20))),
        'min_data': hp.choice('min_data', list(range(100, 301, 20))),
        'min_hessian': hp.loguniform('min_hessian', -1, 0),
        'num_boost_round': hp.choice('num_boost_round', [300, 500, 750]),
        'verbose': -1
    },
    'feature_list': configuration['feature_list']
}

best = fmin(train_wrapper, space, algo=tpe.suggest, max_evals=350)

print(best)
print(space_eval(space, best))
