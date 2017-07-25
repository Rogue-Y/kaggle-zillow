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

        self.predictors = []


    def load_train_test(self, tup):
        self.X_train, self.y_train, self.X_test, self.y_test = tup
        self.X_train_m, self.y_train_m, self.X_test_m, self.y_test_m = list(map(lambda x: x.as_matrix(), tup))

    def check_valid(self):
        assert len(self.X_train_m) > 0
        assert len(self.X_test_m) > 0
        assert len(self.X_train_m) == len(self.y_train_m)
        assert len(self.X_test_m) == len(self.y_test_m)

    def preprocess_target(self, y_train):
        return np.arctan(32.0301149*(-0.005208795+y_train))/3.141593

    def postprocess_target(self, y_predict):
        return 0.0052088+0.0312206*np.tan(3.141593*y_predict)

    def fit(self, predictor, transform_target=False, model_name=''):
        self.check_valid()
        y_train_m_trans = self.y_train_m
        if transform_target:
            print("preprocessing target")
            y_train_m_trans = self.preprocess_target(self.y_train_m)

        print("Fitting from training data")
        predictor.fit(self.X_train_m, y_train_m_trans)
        self.predictors.append({'predictor': predictor, 'transform_target': transform_target})

        print("Predicting")
        y_train_predict = predictor.predict(self.X_train_m)
        y_test_predict = predictor.predict(self.X_test_m)
        if transform_target:
            print("postprocessing target")
            y_train_predict = self.postprocess_target(y_train_predict)
            y_test_predict = self.postprocess_target(y_test_predict)
        print("Results:")
        print(model_name)
        print("Training set", self.mean_error(y_train_predict, self.y_train_m))
        print("Testing set", self.mean_error(y_test_predict, self.y_test_m))

    def mean_error(self, pred, truth):
        return np.mean(abs(pred - truth))
