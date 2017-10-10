import config
from train import *

def load_full_data(config_file):
    # Configuration:
    config_dict = getattr(config, config_file)
    # print configuration for confirmation
    for key, value in config_dict.items():
        if key == 'feature_list':
            for k, v in value.items():
                print('%s: %s' % (k, len(v)))
        elif key == 'stacking_params':
            continue
        else:
            print('%s: %s' % (key, value))

    # Mandatory configurations:
    # Feature list
    df2016, df_all, prop2016, prop2017 = get_dfs(config_dict, True)
    return df2016, df_all, prop2016, prop2017

def train_model(config_file):
    '''
        Returns a tuple like (avg_cv_errors, list of models trained in cross validation)
    '''
    # Configuration:
    config_dict = getattr(config, config_file)
    # print configuration for confirmation
    for key, value in config_dict.items():
        if key == 'feature_list':
            for k, v in value.items():
                print('%s: %s' %(k, len(v)))
        elif key == 'stacking_params' or key == 'tuning_params':
            continue
        else:
            print('%s: %s' %(key, value))

    df2016, df_all, _, _ = get_dfs(config_dict)
    Model = config_dict['Model']
    params = config_dict['training_params']
    return train_process(df2016, df_all, Model, params, 'return_models')
