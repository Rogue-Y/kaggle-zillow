__author__ = 'peter'
import numpy as np

class Evaluator:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        self.X_train_m = []
        self.y_train_m = []
        self.X_test_m = []
        self.y_test_m = []

    def load_train_test(self, tup):
        self.X_train, self.y_train, self.X_test, self.y_test = tup
        self.X_train_m, self.y_train_m, self.X_test_m, self.y_test_m = list(map(lambda x: x.as_matrix(), tup))

    def check_valid(self):
        assert len(self.X_train_m) > 0
        assert len(self.X_train_m) == len(self.y_train_m)
        assert len(self.X_test_m) == len(self.y_test_m)

    def fit(self, predictor):
        self.check_valid()
        print("Fitting from training data")
        predictor.fit(self.X_train_m, self.y_train_m)
        print("Predicting")
        y_train_predict = predictor.predict(self.X_train_m)
        y_test_predict = predictor.predict(self.X_test_m)
        print ("Training set", self.mean_error(y_train_predict, self.y_train_m))
        print ("Testing set", self.mean_error(y_test_predict, self.y_test_m))

    def mean_error(self, pred, truth):
        return np.mean(abs(pred - truth))
