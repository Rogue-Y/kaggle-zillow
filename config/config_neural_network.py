import os
import sys
import sklearn
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_non_linear
# model
from models import NeuralNetwork
# for defining tunning parameters
from hyperopt import hp

# Configuration
config_neural_network = {
    'name': 'config_neural_network',
    'Model': NeuralNetwork.NeuralNet,
    'feature_list': feature_list_non_linear.feature_list_all,
    'clean_na': True,
    'training_params': {
        'model_params': {
            'hidden_layer_sizes': (20, 10),
            'activation': 'relu',
            'solver': 'sgd',
            'alpha': 0.0001,
            'batch_size': 1000,
            # 'learning_rate': 'constant'
            'learning_rate_init': 0.005,
            'tol': 0.01
            'max_iter': 20000,
            # 'beta_1': 0.9,
            # 'beta_2': 0.999
            # 'epsilon':
        },
        'outliers_up_pct': 97,
        'outliers_lw_pct': 5,
        'scaling': True,
        # 'scaler': RobustScaler(quantile_range=(0, 99)),
        # 'scaling_columns': SCALING_COLUMNS
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'alpha': hp.loguniform('alpha', -2, 2),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']),
                'tol': hp.loguniform('tol', -6, -2),
                'random_state': 42
            },
            'outliers_up_pct': 97, # hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': 5,  #hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'scaling': True,
            'pca_components': -1, #hp.choice('pca_components', [-1, 30, 50, 100, 150, 200]),
        },
        'max_evals': 500
    }
}
