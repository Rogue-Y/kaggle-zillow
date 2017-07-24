__author__ = 'peter'
import lightgbm as lgb

class Lightgbm():
    def __init__(self, params):
        self.params = params
        self.model=None

    def fit(self, X, y):
        # self.model = lgb.train(self.params, d_train, num_boost_round=500, valid_sets=[d_valid])

    def predict(self, X):
        pass