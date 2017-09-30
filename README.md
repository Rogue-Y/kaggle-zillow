# kaggle-zillow

May the Randomness be with us
====================

[Data dictionary exploration](https://docs.google.com/spreadsheets/d/1_EHvgdIrkDVPs4p98cPQ26inMz349SIioesTD6B7oHw/edit#gid=1497391001)

[Asana task list](https://app.asana.com/0/389439275300204/board)

The jupyter notebooks are in the notebooks folder. To import our modules correctly in the
notebooks, add the following code at the beginning:

```
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
```

Also, pass `../data` as data folder to the `utils`'s load data functions.

## Introduction

## Feature list

A features list define a dataset models train, validate and predict on.

TODO: elaborate.


## Configuration

All configuration files contains one or more python dictionary variables,
each dictionary represents a configuration.

**(Important!) All configuration dictionary variables need to be added to
`config/__init__.py` for other code to see them.**



### Model configuration

Use `config_linear.py` as an example

Each model configuration dictionary represents a model-feature_list combination.

it has the following keywords:

`name`: name of the configuration, ideally same with the dict variable name

`Model`: the model class

`feature_list`: the feature list

`clean_na (optional)`: if clean nan, default to False. Need to be True for all
models except xgboost and lightgbm. Need to be True when using PCA

`training_params (optional)`: parameters when using train.py to train or
generate submission

`stacking_params (optional)`: parameters when used in stacking, should mostly be
 the same as training_params, consider merging them

`tuning_params (optional)`: parameters when using tune.py to find the best
parameters for the model, include parameter_space and max_evals


### Stacking configuration

Use `stacking_config_test.py` as an example

Stacking configuration takes a list of model configuration as input to stack

TODO: complete this part.


## Train and generate submissions

`train.py` trains models and generate single model submissions.

To train a model and see its cross-validation mae, run:

`python train.py --config model_config_dict_name`

To train and generate a submission:

`python train.py --config model_config_dict_name --submit`

submission will be generated as a csv in `data/submissions` folder, with name
*"Submission_date time"*.


## Stacking

`ensemble.py` is used to train and generate submissions for stacking

To run a stacking see its cross-validation mean average error, run:

`python ensemble.py --stacking --config stacking_config_dict_name`

To stack and generate submissions, run:

`python ensemble.py --stacking --config stacking_config_dict_name --submit`

submission will be generated as a csv in `data/submissions` folder, with name
*"Submission_date time"*.


## Tuning

use `tune.py` to tune model and stacking.

The trials of each run will be saved in `data/trials` (for stacking
`data/trials/stacking`) as pickles, the naming format is
*"modelName_dateTime_pickle"*.

Load them with python pickle, to see all trials, use `trials` property; to see
the best trial, use `best_trial` properties.

[A tutorial of hyperopt, see here](https://github.com/hyperopt/hyperopt/wiki/FMin)


### Tune a single model

Define tuning parameters of a model-feature set in its configuration dict,
`config_linear.py` can be used as an example, tunning parameters (parameter space
and max number evaluations of the hyperopt tuner) should be defined as the value
of key 'tuning_params'.

*Remember to add the configuration dict to `__init__.py` after create it, so that
it can be seen by other code*.

Then, to tune a specific model configuration:

`python tune.py --model --config model_config_dict_name`

Or you can add the some config_dicts to the `model_experiments` list in
`tune.py`, and run the following cmd to train them one by one:

`python tune.py --model`

### Tune stacking

Define a stacking in a stacking configuration dictionary first,
`stacking_config_test.py` can be used as a example.

*Remember to add the configuration dict to `__init__.py` after create it, so that
it can be seen by other code*.

Then, to tune a specific stacking configuration:

`python tune.py --stacking --config stacking_config_dict_name`

Or you can add the some config_dicts to the `stacking_experiments` list in
`tune.py`, and run the following cmd to train them one by one:

`python tune.py --stacking`
