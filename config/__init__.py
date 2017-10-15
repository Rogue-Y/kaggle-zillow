# from .test_config import test_config
from .config_linear import config_linear_huber, config_linearlasso, config_linearridge, config_linearRANSAC
from .config_lightgbm import config_lightgbm, config_lightgbm_geo, config_lightgbm_all_regression_l2, config_lightgbm_all_regression_l1, config_lightgbm_all_huber, config_lightgbm_all_fair, config_lightgbm_all_regression_l1_dart

from .config_xgboost import config_xgboost, config_xgboost_fi, config_xgboost_all, config_xgboost_all_dart
from .config_ensembles import config_rf, config_extra_tree, config_gb
from .config_gaussian_process import config_gaussian_process
from .config_catboost import config_catboost_clean, config_manycatsboost_clean

from .config_neighbors import config_kneighbors


# stacking configurations
from .stacking_config_test import stacking_config_test, stacking_config_linear, stacking_config_ridge
