__author__ = 'peter'
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import GridSearchCV

class Evaluator:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        #
        # self.X_train_m = []
        # self.y_train_m = []
        # self.X_test_m = []
        # self.y_test_m = []

        self.predictors = []


    def load_train_test(self, tup):
        self.X_train, self.y_train, self.X_test, self.y_test = tup
        # self.X_train_m, self.y_train_m, self.X_test_m, self.y_test_m = list(map(lambda x: x.as_matrix(), tup))

    def check_valid(self):
        assert len(self.X_train) > 0
        assert len(self.X_test) > 0
        # assert len(self.X_train_m) == len(self.y_train_m)
        # assert len(self.X_test_m) == len(self.y_test_m)

    def preprocess_target(self, y_train):
        return np.arctan(32.0301149*(-0.005208795+y_train))/3.141593

    def postprocess_target(self, y_predict):
        return 0.0052088+0.0312206*np.tan(3.141593*y_predict)

    def fit(self, predictor, transform_target=False, model_name='', weight=1,
        error_output=1000, save_result=False, predictor_params=None):
        self.check_valid()
        print("Using model: ", model_name)
        y_train_trans = self.y_train
        if transform_target:
            print("preprocessing target")
            y_train_trans = self.preprocess_target(self.y_train)

        print("Fitting from training data")
        predictor.fit(self.X_train, y_train_trans)

        print("Predicting")
        y_train_predict = predictor.predict(self.X_train)
        y_test_predict = predictor.predict(self.X_test)
        if transform_target:
            print("postprocessing target")
            y_train_predict = self.postprocess_target(y_train_predict)
            y_test_predict = self.postprocess_target(y_test_predict)
        # Save the predictor to the predictor list for output submission
        if save_result:
            self.predictors.append({
                'predictor': predictor,
                'model_name':model_name,
                'transform_target': transform_target,
                'weight': weight,
                'y_test_predict': y_test_predict,
                'grid_search': False})

        # output result to console and file (optional)
        print("Results:")
        print(model_name)
        # Get the detail of the prediction, prepare for print to consle and
        # write to record
        train_errors = abs(y_train_predict - self.y_train)
        mean_train_error = Evaluator.mean_error(y_train_predict, self.y_train)
        print("Training set", mean_train_error)
        test_errors = abs(y_test_predict - self.y_test)
        mean_test_error = Evaluator.mean_error(y_test_predict, self.y_test)
        print("Testing set", mean_test_error)
        print("params", predictor_params)
        # Write the data with largest error to a file
        if error_output <= 0:
            return
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_output = pd.concat([self.X_train, self.y_train], axis=1)
        train_output['train_predict'] = y_train_predict
        train_output['train_error'] = train_errors
        test_output = pd.concat([self.X_test, self.y_test], axis=1)
        test_output['test_predict'] = y_test_predict
        test_output['test_error'] = test_errors
        train_output.nlargest(error_output, "train_error").to_csv('data/error/%s_%s_train.csv' %(time, model_name), index=False)
        test_output.nlargest(error_output, "test_error").to_csv('data/error/%s_%s_test.csv' %(time, model_name), index=False)
        with open('data/error/%s_%s_params.txt' %(time, model_name), 'w') as params_output:
            params_output.write(model_name + '\n')
            params_output.write('transform_target: ' + str(transform_target) + '\n')
            params_output.write('grid search: ' + str(False) + '\n')
            params_output.write('params: ' + str(predictor_params) + '\n')
            params_output.write('Train error: ' + str(mean_train_error) + '\n')
            params_output.write('Test error: ' + str(mean_test_error) + '\n')
            params_output.write('\nTrain Stats \n')
            params_output.write('Train label stats: ' + self.y_train.describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('Train predict stats: ' + train_output['train_predict'].describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('Train error stats: ' + train_output['train_error'].describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('\nTest Stats \n')
            params_output.write('Test label stats: ' + self.y_test.describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('Test predict stats: ' + test_output['test_predict'].describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('Test error stats: ' + test_output['test_error'].describe().to_string(float_format='{:.5f}'.format) + '\n')


    def grid_search(self, predictor, param_space, grid_search_params,
        transform_target=False, model_name='', weight=1, error_output=1000,
        save_result=False):
        """ Grid search parameter space for the predictor.
            Returns: best fit predictor based on cross validation.
        """
        print("Using model: ", model_name)
        self.check_valid()
        y_train_trans = self.y_train
        if transform_target:
            print("preprocessing target")
            y_train_trans = self.preprocess_target(self.y_train)

        print("Grid Searching")
        prd = GridSearchCV(predictor, param_space, Evaluator.scorer, **grid_search_params)
        prd.fit(self.X_train, y_train_trans)

        print("Predicting")
        y_train_predict = prd.predict(self.X_train)
        y_test_predict = prd.predict(self.X_test)
        if transform_target:
            print("postprocessing target")
            y_train_predict = self.postprocess_target(y_train_predict)
            y_test_predict = self.postprocess_target(y_test_predict)
        if save_result:
            # Add the cv_predictor to the predictor list for output submission.
            self.predictors.append({
                'predictor': prd,
                'model_name':model_name,
                'transform_target': transform_target,
                'weight': weight,
                'y_test_predict': y_test_predict,
                'grid_search': True})

        # output result to console and file (optional)
        print("Results:")
        print(model_name)
        # Get the detail of the prediction, prepare for print to consle and
        # write to record
        train_errors = abs(y_train_predict - self.y_train)
        mean_train_error = Evaluator.mean_error(y_train_predict, self.y_train)
        print("Training set", mean_train_error)
        test_errors = abs(y_test_predict - self.y_test)
        mean_test_error = Evaluator.mean_error(y_test_predict, self.y_test)
        print("Testing set", mean_test_error)
        print("CV score:", prd.best_score_)
        # print("CV result:", prd.cv_results_)
        print("Best params", prd.best_params_)
        # Write the data with largest error to a file
        if error_output <= 0:
            return
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_output = pd.concat([self.X_train, self.y_train], axis=1)
        train_output['train_predict'] = y_train_predict
        train_output['train_error'] = train_errors
        test_output = pd.concat([self.X_test, self.y_test], axis=1)
        test_output['test_predict'] = y_test_predict
        test_output['test_error'] = test_errors
        train_output.nlargest(error_output, "train_error").to_csv('data/error/%s_%s_train.csv' %(time, model_name), index=False)
        test_output.nlargest(error_output, "test_error").to_csv('data/error/%s_%s_test.csv' %(time, model_name), index=False)
        with open('data/error/%s_%s_params.txt' %(time, model_name), 'w') as params_output:
            params_output.write(model_name + '\n')
            params_output.write('transform_target: ' + str(transform_target) + '\n')
            params_output.write('grid_search: ' + str(True) + '\n')
            params_output.write('best params: ' + str(prd.best_params_) + '\n')
            params_output.write('best scores on cv: ' + str(prd.best_score_) + '\n')
            params_output.write('Train error: ' + str(mean_train_error) + '\n')
            params_output.write('Test error: ' + str(mean_test_error) + '\n')
            params_output.write('\nTrain Stats \n')
            params_output.write('Train label stats: ' + self.y_train.describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('Train predict stats: ' + train_output['train_predict'].describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('Train error stats: ' + train_output['train_error'].describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('\nTest Stats \n')
            params_output.write('Test label stats: ' + self.y_test.describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('Test predict stats: ' + test_output['test_predict'].describe().to_string(float_format='{:.5f}'.format) + '\n')
            params_output.write('Test error stats: ' + test_output['test_error'].describe().to_string(float_format='{:.5f}'.format) + '\n')


    @staticmethod
    def scorer(predictor, X, y):
        return -Evaluator.mean_error(predictor.predict(X), y)

    @staticmethod
    def mean_error(pred, truth):
        return np.mean(abs(pred - truth))
