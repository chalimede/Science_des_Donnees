# pylint: disable=line-too-long

""" params_ml module """

################################################################
## Copyright(C) 2024, Charles Theetten, <chalimede@proton.me> ##
################################################################

from    hyperopt            import hp
from    hyperopt.pyll       import scope

################################################################################

space_lrc       = { "C"                     : hp.uniform("C", 1, 10000),
                    "class_weight"          : hp.choice("class_weight", [ {0:0.15, 1:0.85}, {0:0.20, 1:0.80},
                                                                          {0:0.25, 1:0.75}, {0:0.30, 1:0.70},
                                                                          {0:0.35, 1:0.65}, {0:0.40, 1:0.60} ]),
                    "l1_ratio"              : hp.uniform("l1", 0, 1),
                    "max_iter"              : scope.int(hp.quniform("max_iter", 100, 1000, 1)),
                    "solver"                : hp.choice("solver", [ "lbfgs", "newton-cholesky", "liblinear", "sag", "saga" ]),
                    "tol"                   : hp.uniform("tol", 0.0001, 10)}

space_knn       = { "algorithm"             : hp.choice("algorithm", [ "kd_tree" ]),
                    "metric"                : hp.choice("metric", [ "euclidean", "manhattan" ]),
                    "n_neighbors"           : scope.int(hp.quniform("n_neighbors", 8, 16, 1)),
                    "weights"               : hp.choice("weights", [ "distance" ])}

space_rfc       = { "criterion"             : hp.choice("criterion", ["entropy", "gini"]),
                    "max_depth"             : scope.int(hp.quniform("max_depth", 6, 12, 2)),
                    "max_features"          : scope.int(hp.quniform("max_features", 6, 12, 2)),
                    "min_samples_leaf"      : scope.int(hp.quniform("min_samples_leaf", 2, 8, 2)),
                    "min_samples_split"     : scope.int(hp.quniform("min_samples_split", 2, 6, 2)),
                    "n_estimators"          : scope.int(hp.quniform("n_estimators", 500, 1000, 50)) }

space_lgb       = { "bagging_fraction"      : hp.uniform("bagging_fraction", 0.50, 1.0),
                    "boosting_type"         : hp.choice("boosting_type", [ "gbdt", "dart" ]),
                    "colsample_bytree"      : hp.uniform("colsample_by_tree", 0.25, 0.75),
                    "learning_rate"         : hp.uniform("learning_rate", 0.075, 0.125),
                    "max_depth"             : scope.int(hp.quniform("max_depth", 6, 12, 2)),
                    "min_data_in_leaf"      : scope.int(hp.quniform("min_data_in_leaf", 250, 500, 10)),
                    "n_estimators"          : scope.int(hp.quniform("n_estimators", 500, 1000, 50)),
                    "num_leaves"            : scope.int(hp.quniform("num_leaves", 32, 128, 2)),
                    "objective"             : hp.choice("objective", [ "binary" ]),
                    "path_smooth"           : scope.int(hp.quniform("path_smooth", 32, 64, 2)),
                    "reg_lambda"            : hp.uniform("reg_lambda", 32, 64),
                    "scale_pos_weight"      : hp.uniform("scale_pos_weight", 0.1, 1)}

################################################################################
