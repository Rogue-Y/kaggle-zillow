# base classifiers
import lightgbm as lgb

def train_lgb(X_train, y_train, X_valid, y_valid):
    params = {}
    params['learning_rate'] = 0.002
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = 'mae'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 60
    params['min_data'] = 500
    params['min_hessian'] = 1

    d_train = lgb.Dataset(X_train, label=y_train)
    d_valid = lgb.Dataset(X_valid, label=y_valid)
    return lgb.train(params, d_train, num_boost_round=500, valid_sets=[d_valid])
