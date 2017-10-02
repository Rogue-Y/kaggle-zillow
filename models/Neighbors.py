from abc import ABC # Abstract base class
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

# abstract base class
class NeighborsRegressor(ABC):
    def __init__(self, model_params = None, train_params = None):
        self.model_params = model_params
        self.train_params = train_params

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

# Neighbors regressors
class KNeighbors(NeighborsRegressor):
    def __init__(self, model_params = None, train_params = None):
        NeighborsRegressor.__init__(self, model_params, train_params)
        self.model = KNeighborsRegressor(**self.model_params)

class RadiusNeighbors(NeighborsRegressor):
    def __init__(self, model_params = None, train_params = None):
        NeighborsRegressor.__init__(self, model_params, train_params)
        self.model = RadiusNeighborsRegressor(**self.model_params)
