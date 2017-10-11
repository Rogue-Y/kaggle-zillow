__author__ = 'peter'
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor

class CatBoost():
    def __init__(self, model_params = None, train_params = None):
        self.model_params = model_params
        self.train_params = train_params
        self.model = CatBoostRegressor(**self.model_params)


    def fit(self, X_train, y_train):
        cat_feature_inds = []
        cat_unique_thresh = 1000
        print (X_train.columns)
        for i, c in enumerate(X_train.columns):
            num_uniques = len(X_train[c].unique())
            if num_uniques < cat_unique_thresh \
                    and not 'sqft' in c \
                    and not 'cnt' in c \
                    and not 'tax_difference' in c \
                    and not 'nbr' in c \
                    and not 'room' in c \
                    and not 'number' in c \
                    and not 'mean' in c \
                    and not 'std' in c \
                    and not 'rate' in c \
                    and not 'logerror' in c \
                    and not 'lat_lon_block' in c \
                    and not '_count' in c \
                    and not 'sin_' in c \
                    and not 'cos_' in c \
                    and not 'ratio' in c:
                cat_feature_inds.append(i)
        print("Cat features are: %s" % [X_train.columns[ind] for ind in cat_feature_inds])
        print("Cat features length is: %s" % len(cat_feature_inds))
        X_train.fillna(-999, inplace=True)
        return self.model.fit(X_train, y_train, cat_features=cat_feature_inds)

    def predict(self, X):
        """ Predict on the given X, need to call fit first
            Returns:
                an array of the predict results, has the same rows as X.
        """
        X.fillna(-999, inplace=True)
        return self.model.predict(X)

    def get_params(self):
        return self.model_params if self.model_params is not None else {}

    def get_features_importances(self):
        return None


class ManyCatsBoost():
    def __init__(self, model_params = None, train_params = None):
        self.model_params = model_params
        self.train_params = train_params
        self.n = 5
        self.model = []
        for i in range(self.n):
            self.model_params['random_seed'] = i
            self.model.append(CatBoostRegressor(**self.model_params))

    def fit(self, X_train, y_train):
        X_train.fillna(-999, inplace=True)
        cat_feature_inds = []
        cat_unique_thresh = 1000
        for i, c in enumerate(X_train.columns):
            num_uniques = len(X_train[c].unique())
            if num_uniques < cat_unique_thresh \
                    and not 'sqft' in c \
                    and not 'cnt' in c \
                    and not 'tax_difference' in c \
                    and not 'nbr' in c \
                    and not 'room' in c \
                    and not 'number' in c \
                    and not 'mean' in c \
                    and not 'std' in c \
                    and not 'rate' in c \
                    and not 'logerror' in c \
                    and not 'lat_lon_block' in c \
                    and not '_count' in c \
                    and not 'sin_' in c \
                    and not 'cos_' in c \
                    and not 'ratio' in c:
                cat_feature_inds.append(i)
        print("Cat features are: %s" % [X_train.columns[ind] for ind in cat_feature_inds])
        print("Cat features length is: %s" % len(cat_feature_inds))

        for model in self.model:
            model.fit(X_train, y_train, cat_features=cat_feature_inds)
        return None

    def predict(self, X):
        """ Predict on the given X, need to call fit first
            Returns:
                an array of the predict results, has the same rows as X.
        """
        X.fillna(-999, inplace=True)
        y_pred = 0.0
        for model in self.model:
            y_pred += model.predict(X)
        y_pred /= self.n
        return y_pred

    def get_params(self):
        return None

    def get_features_importances(self):
        return None