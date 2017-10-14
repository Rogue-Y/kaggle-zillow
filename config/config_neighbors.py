import os
import sys
import sklearn
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Feature list
from features import feature_list_neighbors

from models import Neighbors
# for defining tunning parameters
from hyperopt import hp

# Configuration
config_kneighbors = {
    'name': 'config_kneighbors',
    'Model': Neighbors.KNeighbors,
    'feature_list': feature_list_neighbors.feature_list,
    'clean_na': True,
    'training_params': {
        'model_params': {
		'leaf_size': 350,
		'n_jobs': -1,
		'n_neighbors': 200,
		'p': 1,
		'weights': 'uniform'
	},
	'outliers_lw_pct': 3,
	'outliers_up_pct': 97,
	'scaling': True
    },
    'tuning_params': {
        'parameter_space': {
            'model_params': {
                'n_neighbors':  hp.choice('n_neighbors', [200, 300, 350]),
                'weights':  hp.choice('weights', ['uniform', 'distance']),
                'leaf_size':  hp.choice('leaf_size', [350, 500, 750]),
                'p':  hp.choice('p', [1, 2]),
                'n_jobs': -1
            },
            'outliers_up_pct': hp.choice('outliers_up_pct', [97, 98, 99]),
            'outliers_lw_pct': hp.choice('outliers_lw_pct', [3, 2, 1]),
            'scaling': True,
        },
        'max_evals': 60
    }
}
