__author__ = 'peter'
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor

class CatBoost():
    def __init__(self, model_params = None, train_params = None):
        self.model_params = model_params
        self.train_params = train_params
        self.model = CatBoostRegressor(**self.model_params)

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
        return None


class ManyCatsBoost():
    def __init__(self, model_params = None, train_params = None):
        self.model_params = model_params
        self.train_params = train_params
        self.n = self.train_params['n_models']
        self.model = []
        for i in range(self.n):
            self.model_params['random_seed'] = i
            self.model.append(CatBoostRegressor(**self.model_params))

    def fit(self, X_train, y_train):
        for model in self.model:
            model.fit(X_train, y_train)
        return None

    def predict(self, X):
        """ Predict on the given X, need to call fit first
            Returns:
                an array of the predict results, has the same rows as X.
        """
        y_pred = 0.0
        for model in self.model:
            y_pred += model.predict(X)
        y_pred /= self.n
        return y_pred

    def get_params(self):
        return None

    def get_features_importances(self):
        return None