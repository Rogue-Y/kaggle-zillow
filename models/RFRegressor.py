__author__ = 'peter'
from sklearn.ensemble import RandomForestRegressor

class RFRegressor():
    def __init__(self, model_params = None, train_params = None):
        # if model_params is None:
        #     model_params = {
        #         'eta': 0.037,
        #         'max_depth': 1,
        #         'subsample': 0.80,
        #         'objective': 'reg:linear',
        #         'eval_metric': 'mae',
        #         'lambda': 0.8,
        #         'alpha': 0.4,
        #         # 'base_score': y_mean,
        #         'silent': 1
        #     }
        # if train_params is None:
        #     train_params = {
        #         'num_boost_round': 250
        #     }
        self.model_params = model_params
        self.train_params = train_params
        self.model = RandomForestRegressor(**dict(self.model_params))

    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X):
        """ Predict on the given X, need to call fit first
            Returns:
                an array of the predict results, has the same rows as X.
        """
        return self.model.predict(X)

    def get_params(self):
        return self.model_params if self.model_params is not None else {}

    def get_features_importances(self):
        pass
