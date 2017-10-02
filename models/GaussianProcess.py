from sklearn.gaussian_process import GaussianProcessRegressor

class GaussianProcess():
    def __init__(self, model_params = None, train_params = None):
        self.model_params = model_params
        self.train_params = train_params
        self.model = GaussianProcessRegressor(**self.model_params)

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
    
