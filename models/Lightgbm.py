__author__ = 'peter'
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

class Lightgbm():
    def __init__(self, model_params = None, data_params = None):
        # TODO(hzn): check params one by one
        if model_params is None:
            model_params = {}
            model_params['learning_rate'] = 0.002
            model_params['boosting_type'] = 'gbdt'
            model_params['objective'] = 'regression'
            model_params['metric'] = 'mae'
            model_params['sub_feature'] = 0.5
            model_params['num_leaves'] = 60
            model_params['min_data'] = 500
            model_params['min_hessian'] = 1
            model_params['num_boost_round'] = 500
            model_params['verbose'] = -1
        if data_params is None:
            data_params = {}
            data_params['test_size'] = 0.25
            data_params['random_state'] = 42
        self.model_params = model_params
        self.data_params = data_params
        self.model=None

    def fit(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size=self.data_params['test_size'],
            random_state=self.data_params['random_state'])

        d_train = lgb.Dataset(X_train, label=y_train)
        d_valid = lgb.Dataset(X_valid, label=y_valid)
        self.model = lgb.train(self.model_params, d_train,
            num_boost_round=self.model_params['num_boost_round'],
            valid_sets=[d_valid])

    def predict(self, X):
        """ Predict on the given X, need to call fit first
            Returns:
                an array of the predict results, has the same rows as X.
        """
        return self.model.predict(X)

class Lightgbm_sklearn():
    def __init__(self, model_params = None):
        # TODO(hzn): check params one by one
        if model_params is None:
            model_params = {
                 "seed": 42, "nthread": 4, "silent": True, "boosting_type": "gbdt",
                 "objective": "regression_l2", "colsample_bytree": 0.7,
                 "learning_rate": 0.03, "max_bin": 30, "min_child_samples": 500,
                 "n_estimators": 30, "reg_lambda": 1, "subsample": 0.7,
                 "subsample_freq": 30
            }
        self.model_params = model_params
        self.model = LGBMRegressor(**model_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """ Predict on the given X, need to call fit first
            Returns:
                an array of the predict results, has the same rows as X.
        """
        return self.model.predict(X)
