from abc import ABC # Abstract base class
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, TheilSenRegressor, RANSACRegressor
#TODO: add other regressions in sklearn linear model module


# abstract base class
class LinearRegressor(ABC):
    def __init__(self, model_params = None, train_params = None):
        self.model_params = model_params
        self.train_params = train_params

    def fit(self, X_train, y_train):
        self.features = X_train.columns
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
        feature_importances = list(zip(self.features, self.model.coef_))
        return sorted(feature_importances, key=lambda x: -x[1])


class RidgeRegressor(LinearRegressor):
    def __init__(self, model_params = None, train_params = None):
        LinearRegressor.__init__(self, model_params, train_params)
        self.model = Ridge(**self.model_params)


class LassoRegressor(LinearRegressor):
    def __init__(self, model_params = None, train_params = None):
        LinearRegressor.__init__(self, model_params, train_params)
        self.model = Lasso(**self.model_params)


class ElasticNetRegressor(LinearRegressor):
    def __init__(self, model_params = None, train_params = None):
        LinearRegressor.__init__(self, model_params, train_params)
        self.model = ElasticNet(**self.model_params)

class Huber(LinearRegressor):
    def __init__(self, model_params = None, train_params = None):
        LinearRegressor.__init__(self, model_params, train_params)
        self.model = HuberRegressor(**self.model_params)

class TheilSen(LinearRegressor):
    def __init__(self, model_params = None, train_params = None):
        LinearRegressor.__init__(self, model_params, train_params)
        self.model = TheilSenRegressor(**self.model_params)

class RANSAC(LinearRegressor):
    def __init__(self, model_params = None, train_params = None):
        LinearRegressor.__init__(self, model_params, train_params)
        self.model = RANSACRegressor(**self.model_params)