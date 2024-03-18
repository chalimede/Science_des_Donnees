# pylint: disable=line-too-long
# pylint: disable=W0012
# pylint: disable=S3776

""" ml_models meteo module """

################################################################
## Copyright(C) 2024, Charles Theetten, <chalimede@proton.me> ##
################################################################

################################################################################

from    functools               import partial

from    hyperopt                import STATUS_OK
from    hyperopt                import fmin, tpe, Trials

from    sklearn.model_selection import StratifiedKFold
from    sklearn.model_selection import cross_val_score
from    sklearn.model_selection import train_test_split
from    sklearn.preprocessing   import StandardScaler

from    sklearn.linear_model    import LogisticRegression
from    sklearn.neighbors       import KNeighborsClassifier
from    sklearn.ensemble        import RandomForestClassifier
from    lightgbm                import LGBMClassifier

from    imblearn.metrics        import classification_report_imbalanced
from    imblearn.over_sampling  import SMOTE
from    imblearn.under_sampling import ClusterCentroids

import  numpy                   as np
import  pandas                  as pd

################################################################################

class MlModels:
    """ ModelMeteo class """

    def __init__(self):
        """ constructor """
        self.data           = None
        self.x              = None
        self.y              = None
        self.x_train        = None
        self.y_train        = None
        self.x_test         = None
        self.y_test         = None

    def init_data(self, src, date_inf, date_sup):
        """ init data """
        self.data                  = pd.read_csv(src)
        self.data["Date"]          = pd.to_datetime(self.data["Date"])
        self.data["RainTomorrow"]  = self.data["RainTomorrow"].astype(np.int8)
        self.data                  = self.data[(self.data["Date"].dt.year >= date_inf) & (self.data["Date"].dt.year <= date_sup)]
        self.data                  = self.data.sort_values(by = ["Date"])
        self.data                  = self.data.drop(["Date", "Location"], axis = 1)

    def eval_model(self, clf, params):
        """ evaluate model """
        train_score = clf.score(self.x_train, self.y_train)
        test_score  = clf.score(self.x_test, self.y_test)
        y_pred      = clf.predict(self.x_test)
        crosstab    = pd.crosstab(self.y_test, y_pred)
        report      = classification_report_imbalanced(self.y_test, y_pred)
        return params, train_score, test_score, crosstab, report

    def search_model(self, space, name):
        """ find hyperparameters"""
        def objective_function(params, name):
            clf = None
            if name == "LRC":
                clf = LogisticRegression(**params, n_jobs = -1)
            if name == "KNN":
                clf = KNeighborsClassifier(**params, n_jobs = -1)
            if name == "RFC":
                clf = RandomForestClassifier(**params, n_jobs = -1)
            if name == "LGB":
                clf = LGBMClassifier(**params)
            skf     = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 123)
            score   = cross_val_score(clf, self.x_train, self.y_train, cv = skf, n_jobs = -1).mean()
            return { "loss" : -score, "status" : STATUS_OK }

        def build_model(params, name):
            if name == "LRC":
                clf = self.build_lrc(params)
            if name == "KNN":
                clf = self.build_knn(params)
            if name == "RFC":
                clf = self.build_rfc(params)
            if name == "LGB":
                clf = self.build_lgb(params)
            return clf

        trials          = Trials()
        fmin_objective  = partial(objective_function, name = name)
        best_params     = fmin(fn = fmin_objective, space = space, algo = tpe.suggest, max_evals = 10, trials = trials)
        values_params   = list(best_params.values())
        clf             = build_model(values_params, name)
        print(best_params)
        return clf, best_params

    def build_lrc(self, params):
        """ build lrc model """
        class_weight    = [ {0:0.15, 1:0.85}, {0:0.20, 1:0.80},
                            {0:0.25, 1:0.75}, {0:0.30, 1:0.70},
                            {0:0.35, 1:0.65}, {0:0.40, 1:0.60} ]
        solver          = [ "lbfgs", "liblinear", "newton-cholesky", "sag", "saga" ]
        clf             = LogisticRegression(C              = params[0],
                                             class_weight   = class_weight[params[1]],
                                             l1_ratio       = params[2],
                                             max_iter       = int(params[3]),
                                             solver         = solver[params[4]],
                                             tol            = params[5])
        clf.fit(self.x_train, self.y_train)
        return clf

    def build_knn(self, params):
        """ build knn model """
        algorithms  = [ "kd_tree" ]
        metrics     = [ "euclidean", "manhattan" ]
        weights     = [ "distance" ]
        clf         = KNeighborsClassifier(algorithm    = algorithms[params[0]],
                                           metric       = metrics[params[1]],
                                           n_neighbors  = int(params[2]),
                                           weights      = weights[params[3]],
                                           n_jobs       = -1)
        clf.fit(self.x_train, self.y_train)
        return clf

    def build_rfc(self, params):
        """ build rfc model """
        criterion   = [ "entropy", "gini" ]
        clf         = RandomForestClassifier(criterion          = criterion[params[0]],
                                             max_depth          = int(params[1]),
                                             max_features       = int(params[2]),
                                             min_samples_leaf   = int(params[3]),
                                             min_samples_split  = int(params[4]),
                                             n_estimators       = int(params[5]),
                                             n_jobs = -1)
        clf.fit(self.x_train, self.y_train)
        return clf

    def build_lgb(self, params):
        """ build lgb model """
        boosting        = [ "gbdt", "dart" ]
        objective       = [ "binary" ]
        clf             = LGBMClassifier(bagging_fraction   = params[0],
                                         boosting_type      = boosting[params[1]],
                                         colsample_bytree   = params[2],
                                         learning_rate      = params[3],
                                         max_depth          = int(params[4]),
                                         min_data_in_leaf   = int(params[5]),
                                         n_estimators       = int(params[6]),
                                         num_leaves         = int(params[7]),
                                         objective          = objective[params[8]],
                                         path_smooth        = params[9],
                                         reg_lambda         = int(params[10]),
                                         scale_pos_weight   = params[11])
        clf.fit(self.x_train, self.y_train)
        return clf

    def over_sample_data(self, strategy):
        """ resample data """
        smo                         = SMOTE(sampling_strategy = strategy, random_state = 123)
        self.x_train, self.y_train  = smo.fit_resample(self.x_train, self.y_train)

    def under_sample_data(self, strategy, voting):
        """ resample data """
        cc                          = ClusterCentroids(sampling_strategy = strategy, voting = voting, random_state = 123)
        self.x_train, self.y_train  = cc.fit_resample(self.x_train, self.y_train)

    def scale_data(self):
        """ scale data """
        scaler          = StandardScaler()
        self.x_train    = scaler.fit_transform(self.x_train)
        self.x_test     = scaler.transform(self.x_test)

    def split_data(self, target, size):
        """ split data """
        self.x = self.data.drop(target, axis = 1)
        self.y = self.data[target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size = size,
                                                                                shuffle = False,
                                                                                random_state = 123)
