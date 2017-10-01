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
