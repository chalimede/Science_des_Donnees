# pylint: disable=line-too-long
# pylint: disable=W0108

""" dl_models module """

################################################################
## Copyright(C) 2024, Charles Theetten, <chalimede@proton.me> ##
################################################################

################################################################################

import  io
import  numpy                   as np
import  pandas                  as pd

from    sklearn.preprocessing   import StandardScaler

from    sklearn.metrics         import classification_report
from    sklearn.metrics         import confusion_matrix

from    keras_tuner             import Hyperband

################################################################################

class DLModels:
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

    def eval_dense_model(self, model, best_params, epochs, split):
        """ evaluate model """
        batch_size = best_params["batch_size"]

        model.fit(self.x_train, self.y_train,
                  epochs            = epochs,
                  batch_size        = batch_size,
                  validation_split  = split,
                  shuffle           = False)

        test_score      = model.evaluate(self.x_test, self.y_test)[1]
        test_pred       = model.predict(self.x_test)
        y_pred_class    = np.where(test_pred > 0.5, 1, 0)
        c_matrix        = confusion_matrix(self.y_test, y_pred_class)
        report          = classification_report(self.y_test, y_pred_class)

        s_summary       = io.StringIO()
        model.summary(print_fn = lambda x : s_summary.write(x + "\n"))
        summary         = s_summary.getvalue()
        s_summary.close()
        return summary, best_params, test_score, c_matrix, report

    def eval_recurrent_model(self, model, best_params, epochs, split):
        """ evaluate model """
        batch_size = best_params["batch_size"]

        model.fit(self.x_train, self.y_train,
                  epochs            = epochs,
                  batch_size        = batch_size,
                  validation_split  = split,
                  shuffle           = False)

        test_score      = model.evaluate(self.x_test, self.y_test)[1]
        test_pred       = model.predict(self.x_test)
        test_pred       = test_pred[:, 0, :]
        y_pred_class    = np.where(test_pred > 0.5, 1, 0)
        c_matrix        = confusion_matrix(self.y_test, y_pred_class)
        report          = classification_report(self.y_test, y_pred_class)

        s_summary       = io.StringIO()
        model.summary(print_fn = lambda x : s_summary.write(x + "\n"))
        summary         = s_summary.getvalue()
        s_summary.close()
        return summary, best_params, test_score, c_matrix, report

    def search_model(self, model, epochs, project):
        """ search model """
        tuner = Hyperband(hypermodel    = model,
                          objective     = "val_accuracy",
                          overwrite     = True,
                          max_epochs    = epochs,
                          project_name  = project)

        tuner.search(self.x_train, self.y_train,
                     epochs             = epochs,
                     validation_data    = (self.x_test, self.y_test))

        best_hyparameters   = tuner.get_best_hyperparameters()[0].values
        best_model          = tuner.get_best_models()[0]
        return best_model, best_hyparameters

    def scale_data(self):
        """ scale data """
        scaler          = StandardScaler()
        self.x_train    = scaler.fit_transform(self.x_train)
        self.x_test     = scaler.transform(self.x_test)

    def split_data(self, target):
        """ split data """
        self.x_train = self.data[self.data["Date"].dt.year < 2016].drop(target, axis = 1)
        self.y_train = self.data[self.data["Date"].dt.year < 2016][target]
        self.x_test  = self.data[self.data["Date"].dt.year >= 2016].drop(target, axis = 1)
        self.y_test  = self.data[self.data["Date"].dt.year >= 2016][target]
        self.x_train = self.x_train.drop(["Date", "Location"], axis = 1)
        self.x_test  = self.x_test.drop(["Date", "Location"], axis = 1)
