# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
#
# # Configuration:
# # folds number of K-Fold
# FOLDS = 5
#
# # Feature list
# from features import feature_list_non_linear
# feature_list = feature_list_non_linear.feature_list
#
# # model
# from models import XGBoost, Lightgbm
# Model = XGBoost.XGBoost
# # Model = Lightgbm.Lightgbm_sklearn
# # Model = Lightgbm.Lightgbm
# # Model = RFRegressor.RFRegressor
#
#
# submit = False
#
# record = False
#
# test_config = {
#     # 'pca_components': 15, # a pca_component greater than 0 will automatically set clean_na to True as pca cannot deal with infinite numbers.
#     # 'resale_offset': 0,
#     'folds': FOLDS,
#     'feature_list': feature_list,
#     'model': Model,
#     # Best lightgbm param with geo_neighborhood and geo_zip, 0.0646722050526
#     # 'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.012923556870735842,
#     #     'metric': 'mse', 'min_data': 200, 'min_hessian': 0.232705809294419,
#     #     'num_boost_round': 500, 'num_leaves': 30, 'objective': 'regression',
#     #     'sub_feature': 0.1596488603529622, 'verbose': -1},
#     # Best lightgbm param with almost all features, 0.0646671968399
#     # 'model_params': {'boosting_type': 'gbdt', 'learning_rate': 0.05525896243523381,
#     #     'metric': 'mae', 'min_data': 260, 'min_hessian': 0.57579034653711,
#     #     'num_boost_round': 300, 'num_leaves': 70, 'objective': 'regression_l1',
#     #     'sub_feature': 0.06638755200543586, 'verbose': -1},
#     # rf parameters
#     # 'model_params': {'max_features': 0.2, 'max_leaf_nodes': None,
#     #         'min_samples_leaf': 70, 'n_estimators': 50},
#     # 'clean_na': True,
#     'submit': submit,
#     'record': record,
#
#     # # Best xgboost with 5 added features + precise geo filling: 0647672681377, lb: 0.0644668
#     # 'model_params': {'alpha': 0.3, 'colsample_bylevel': 0.3,
#     #     'colsample_bytree': 0.5, 'eta': 0.07455450922244707,
#     #     'eval_metric': 'mae', 'gamma': 8.249459830776771e-05, 'lambda': 0.6,
#     #     'max_depth': 4, 'min_child_weight': 0.9055707037083442,
#     #     'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6},
#     # 'outliers_lw_pct': 2,
#     # 'outliers_up_pct': 98
#
#     # xgboost with 5 added features + target features: 0.0644820716823
#     # 'model_params': {'alpha': 0.0, 'colsample_bylevel': 0.5,
#     #     'colsample_bytree': 0.6, 'eta': 0.057797411022032265,
#     #     'eval_metric': 'rmse', 'gamma': 0.026493673908889032, 'lambda': 0.7,
#     #     'max_depth': 6, 'min_child_weight': 1.913900160701277,
#     #     'objective': 'reg:linear', 'silent': 1, 'subsample': 0.8},
#     # 'outliers_lw_pct': 4, 'outliers_up_pct': 97
#
#     # Based on above, with target: 0646927
#     # Add std/mean ratio and range, fillna: 0646785
#     # 'model_params': {'alpha': 0.3, 'colsample_bylevel': 0.3,
#     #     'colsample_bytree': 0.5, 'eta': 0.07455450922244707,
#     #     'eval_metric': 'mae', 'gamma': 8.249459830776771e-05, 'lambda': 0.6,
#     #     'max_depth': 4, 'min_child_weight': 0.9055707037083442,
#     #     'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6},
#     # 'outliers_lw_pct': 2,
#     # 'outliers_up_pct': 98,
#
#     # lightgbm: cv 0.064637416215, lb:0.0645447
#     # 'model_params': {'bagging_fraction': 0.8384638110940468, 'bagging_freq': 0,
#     #     'boosting_type': 'gbdt', 'learning_rate': 0.1353711356306096,
#     #     'max_bin': 100, 'metric': 'mse', 'min_data': 230,
#     #     'min_hessian': 0.5961775594444781, 'num_boost_round': 200, 'num_leaves': 50,
#     #     'objective': 'regression', 'sub_feature': 0.18462105358643505, 'verbose': -1},
#     # 'outliers_lw_pct': 4, 'outliers_up_pct': 96
#
#     # xgboost with almost all features + precise geo filling: cv 0.0646354220721, lb: 0.0643522
#     # same parameter with 5 features + target features + precise geo filling: cv 0.0645577144428, lb: 0.0643933
#     # with almost all features + target features + precise geo filling: cv 0.0646190894623, lb: 0.0643531
#     # 'model_params': { 'alpha': 0.6, 'colsample_bylevel': 0.7, 'colsample_bytree': 0.7,
#     #     'eta': 0.08383948785330207, 'eval_metric': 'rmse', 'gamma': 0.001115761304103735,
#     #     'lambda': 0.4, 'max_depth': 4, 'min_child_weight': 4.092393060805701, 'subsample': 0.6},
#     # 'outliers_lw_pct': 4, 'outliers_up_pct': 97
#
#     # further tuning of the above one: cv 0.0646223502383, lb: 0.0643487
#     # changed order of features (put missing values first): cv 0.0647246470924, lb: 0.0643368
#     # plus target: cv: 0.0645315564755 lb: 0.0643622
#     'model_params': {'alpha': 0.6, 'colsample_bylevel': 0.7, 'colsample_bytree': 0.7, 'eta': 0.07901772316032044,
#         'eval_metric': 'rmse', 'gamma': 0.0018188912716341973, 'lambda': 0.4, 'max_depth': 4,
#         'min_child_weight': 4.4156043204121, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.6},
#     'outliers_lw_pct': 4, 'outliers_up_pct': 97
#
#     # xgboost with almost all features + precise geo filling + target features: cv 0.064534428294, lb: 0.0643728
#     # 'model_params': {'alpha': 0.4, 'colsample_bylevel': 0.5, 'colsample_bytree': 0.5, 'eta': 0.13806545489668282,
#     #     'eval_metric': 'rmse', 'gamma': 0.010959418042539222, 'lambda': 0.0, 'max_depth': 3,
#     #     'min_child_weight': 5.990179308552547, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.8},
#     # 'outliers_lw_pct': 4, 'outliers_up_pct': 97
#
#     # xgboost with almost all features + precise geo filling + target features: cv 0.0646297759966, lb: 0.0643844
#     # 'model_params': {
#     #     'alpha': 0.1, 'colsample_bylevel': 0.5, 'colsample_bytree': 0.8, 'eta': 0.143998802451337,
#     #     'eval_metric': 'rmse', 'gamma': 0.053876541022518674, 'lambda': 0.0, 'max_depth': 7,
#     #     'min_child_weight': 2.6616870580729772, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.8},
#     # 'outliers_lw_pct': 4, 'outliers_up_pct': 97
# }
