# Ensembling prediction result
# TODO(hzn):
#   *1. how to record ensembling experiements result
#   *2. how to envaluate ensembling in cv rather than submit the result, predictions
#       actually contains the training houses
#   *3. following the above one, how to automatically tune ensembling
#   *3. adjust for resale first or ensembling first (mathematically the same,
#       but the adjust offset could be different for different ensembling, also
#       the adjustment may affect validation, validation should be made with
#       pre-adjusted predictions)

import datetime
import gc
import time
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import config_linear, lightgbm_config
from evaluator import Evaluator
from features import utils, data_clean
from models import Lightgbm
from train import prepare_features, prepare_training_data, train_stacking

# tuples of format (pickle_path, weight)
prediction_list = [
    # ('XGBoost_latest_pickle', 7),
    ('Lightgbm_latest_pickle', 3)
]

def ensemble(prediction_list=prediction_list):
    print('Ensembling...')
    print('loading sample submission...')
    df_test, sample = utils.load_test_data()
    # ensembling
    print('ensembling...')
    # initialize the prediction with bias, currently 0
    pred_sum = np.zeros(sample.shape[0])
    total_weight = 0
    for pickle_path, weight in prediction_list:
        pred = pd.read_pickle('data/predictions/%s' %pickle_path)
        pred = weight * pred.as_matrix()
        total_weight += weight
        pred_sum += pred
    ensembled_pred = pred_sum / total_weight

    # generate submission
    print("generating submission...")
    for c in sample.columns[sample.columns != 'ParcelId']:
        # sample[c] = avg_pred
        sample[c] = ensembled_pred
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sample.to_csv(
        'data/submissions/Submission_%s.csv' %time, index=False, float_format='%.4f')

def get_first_layer(stacking_list):
    print('Generate first layer...')

    first_layer_preds = []

    for config_dict in stacking_list:
        # Read config
        # Mandatory configurations:
        # Feature list
        feature_list = config_dict['feature_list']
        # clean_na
        clean_na = config_dict['clean_na'] if 'clean_na' in config_dict else False

        prop = prepare_features(feature_list, clean_na)
        train_df, transactions = prepare_training_data(prop)
        del transactions; del prop; gc.collect()

        validation_pred, validation_target, _ = train_stacking(train_df, **config_dict['stacking_params'])
        first_layer_preds.append(validation_pred)

    # assemble first layer result
    first_layer = pd.concat(first_layer_preds, axis=1)
    return first_layer, validation_target

def stacking(first_layer, target, meta_model):
    print(first_layer.shape)
    print(target.shape)
    assert len(first_layer) == len(target)

    print('second layer...')
    mean_errors = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # same k fold split as above
    # (this is guaranteed as long as the random seeds and the number of rows does not change,
    # therefore we only need to make sure the indexes of the dataframes are always aligned)
    for train_index, validate_index in kf.split(first_layer):
        X_train, y_train = first_layer.loc[train_index], target.loc[train_index]
        X_validate, y_validate = first_layer.loc[validate_index], target.loc[validate_index]

        print('training...')
        meta_model.fit(X_train, y_train)
        train_pred = meta_model.predict(X_train)
        print("fold train mean error: ", Evaluator.mean_error(train_pred, y_train))

        print('validating...')
        y_pred = meta_model.predict(X_validate)
        # TODO(hzn): add output training records
        mae = Evaluator.mean_error(y_pred, y_validate)
        mean_errors.append(mae)
        print("fold validation mean error: ", mae)

    avg_cv_errors = np.mean(mean_errors)
    print("average cross validation mean error", avg_cv_errors)
    return avg_cv_errors

# Main method
if __name__ == '__main__':
    t1 = time.time()
    # Get configuration
    # parser to parse cmd line option
    parser = OptionParser()
    # ensemble is true by default or when -e flag is present
    parser.add_option('-e', '--ensemble', action='store_true', dest='ensemble', default=True)
    # ensemble is set to false when -s flag is present
    parser.add_option('-s', '--stacking', action='store_false', dest='ensemble')
    # parse cmd line arguments
    (options, args) = parser.parse_args()

    if options.ensemble:
        ensemble()
    else:
        # stacking
        stacking_list = [
            config_linear,
            lightgbm_config
        ]
        meta_model = Lightgbm.Lightgbm()

        first_layer, first_layer_target = get_first_layer(stacking_list)
        stacking(first_layer, first_layer_target, meta_model)

    t2 = time.time()
    print((t2 - t1) / 60)
