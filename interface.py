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
    feature_list = config_dict['feature_list']
    clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False

    prop = prepare_features(feature_list, clean_na)
    train_df, transactions = prepare_training_data(prop)
    return train_df, prop, transactions

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

    # Mandatory configurations:
    # Feature list
    feature_list = config_dict['feature_list']
    # # model
    Model = config_dict['Model']
    # clean_na
    clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False

    prop = prepare_features(feature_list, clean_na)

    train_df, transactions = prepare_training_data(prop)
    del transactions; del prop; gc.collect()
    return train(train_df, Model=Model, submit=False, return_models=True, **config_dict['training_params'])
