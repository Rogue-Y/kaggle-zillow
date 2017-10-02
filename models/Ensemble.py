from abc import ABC # Abstract base class
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

# abstract base class
class EnsembleRegressor(ABC):
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
        feature_importances = list(zip(self.features, self.model.feature_importances_))
        return sorted(feature_importances, key=lambda x: -x[1])


# Ensemble regressors
class AdaBoost(EnsembleRegressor):
    def __init__(self, model_params = None, train_params = None):
        EnsembleRegressor.__init__(self, model_params, train_params)
        self.model = AdaBoostRegressor(**self.model_params)


class Bagging(EnsembleRegressor):
    def __init__(self, model_params = None, train_params = None):
        EnsembleRegressor.__init__(self, model_params, train_params)
        self.model = BaggingRegressor(**self.model_params)


class ExtraTrees(EnsembleRegressor):
    def __init__(self, model_params = None, train_params = None):
        EnsembleRegressor.__init__(self, model_params, train_params)
        self.model = ExtraTreesRegressor(**self.model_params)


class GradientBoosting(EnsembleRegressor):
    def __init__(self, model_params = None, train_params = None):
        EnsembleRegressor.__init__(self, model_params, train_params)
        self.model = GradientBoostingRegressor(**self.model_params)


class RandomForest(EnsembleRegressor):
    def __init__(self, model_params = None, train_params = None):
        EnsembleRegressor.__init__(self, model_params, train_params)
        self.model = RandomForestRegressor(**self.model_params)
