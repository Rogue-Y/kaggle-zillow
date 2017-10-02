import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_linear
# model
from models import GaussianProcess
# for defining tunning parameters
from hyperopt import hp

# Configuration
config_gaussian_process = {
    'name': 'config_gaussian_process',
    'Model': GaussianProcess.GaussianProcess,
    'feature_list': feature_list_linear.feature_list,
    'clean_na': True,
    'training_params': {
    },
    'stacking_params': {
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                #TODO: investigate more kernels
                'alpha': hp.loguniform('alpha', -10, -1),
                'n_restarts_optimizer': hp.choice('n_restarts_optimizer', [0, 5, 10]),
                'normalize_y': hp.choice('normalize_y', [True, False]),
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [95, 96, 97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [5, 4, 3, 2, 1]),
            'scaling': True,
            'pca_components': hp.choice('pca_components', [-1, 30, 50, 100, 150, 200]),
        },
        'max_evals': 2
    }
}
