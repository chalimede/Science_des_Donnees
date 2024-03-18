# pylint: disable=line-too-long

""" results module """

################################################################
## Copyright(C) 2024, Charles Theetten, <chalimede@proton.me> ##
################################################################

from joblib         import dump

class Results:
    """ Results class """

    def __init__(self):
        """ constructor """
        self.models         = { }

    def init_models(self, name):
        """ init dictionary models """
        self.models[name]   = { }

    def register_ml(self, name, clf, results):
        """ register model """
        self.models[name]["clf"]    = clf
        self.models[name]["params"] = results[0]
        self.models[name]["trs"]    = results[1]
        self.models[name]["tes"]    = results[2]
        self.models[name]["cm"]     = results[3]
        self.models[name]["cr"]     = results[4]

    def register_dl(self, name, clf, results):
        """ register model """
        self.models[name]["clf"]        = clf
        self.models[name]["summary"]    = results[0]
        self.models[name]["params"]     = results[1]
        self.models[name]["tes"]        = results[2]
        self.models[name]["cm"]         = results[3]
        self.models[name]["cr"]         = results[4]

    def persist_dl_model(self, clf, name, path, df):
        """ persist model """
        print(path + name + df + ".keras")
        clf.save(path + name + df + ".keras")

    def persist_ml_model(self, name, path, df):
        """ persist model """
        file = path + name + df + ".joblib"
        dump(self.models[name]["clf"], file)

    def print_dl_results(self, name):
        """ write results on metrics """
        model       = f"Model                   : {name}\n"
        summary     = f"Summary                 :\n{self.models[name]['summary']}\n"
        test_score  = f"Test score              : {self.models[name]['tes']:.3f}\n"
        cf_matrix   = f"Confusion matrix        :\n\n{self.models[name]['cm']}\n\n"
        report      = f"Classification report   :\n{self.models[name]['cr']}\n"
        delimiter   = "#" * 100 + "\n\n"
        results     = model + summary + test_score + cf_matrix + report + delimiter
        print(results)

    def print_ml_results(self, name):
        """ write results on metrics """
        model       = f"Model                   : {name}\n"
        params      = f"Hyperparameters         : {self.models[name]['params']}\n"
        train_score = f"Train Score             : {self.models[name]['trs']:.3f}\n"
        test_score  = f"Test Score              : {self.models[name]['tes']:.3f}\n"
        cf_matrix   = f"Confusion matrix        :\n\n{self.models[name]['cm']}\n\n"
        report      = f"Classification report   :\n{self.models[name]['cr']}\n"
        delimiter   = "#" * 100 + "\n\n"
        results     = model + params + train_score + test_score + cf_matrix + report + delimiter
        print(results)

    def write_dl_results(self, name, filename):
        """ write results on metrics """
        model       = f"Model                   : {name}\n"
        summary     = f"Summary                 :\n{self.models[name]['summary']}\n"
        params      = f"Hyperparameters         :\n{self.models[name]['params']}\n"
        test_score  = f"Test score              : {self.models[name]['tes']:.3f}\n"
        cf_matrix   = f"Confusion matrix        :\n\n{self.models[name]['cm']}\n\n"
        report      = f"Classification report   :\n{self.models[name]['cr']}\n"
        delimiter   = "#" * 100 + "\n\n"
        results     = model + summary + params + test_score + cf_matrix + report + delimiter
        with open(filename, "a+", encoding = "utf-8") as file:
            file.write(results)
            file.close()

    def write_ml_results(self, name, filename):
        """ write results on metrics """
        model       = f"Model                   : {name}\n"
        params      = f"Hyperparameters         : {self.models[name]['params']}\n"
        train_score = f"Train Score             : {self.models[name]['trs']:.3f}\n"
        test_score  = f"Test Score              : {self.models[name]['tes']:.3f}\n"
        cf_matrix   = f"Confusion matrix        :\n\n{self.models[name]['cm']}\n\n"
        report      = f"Classification report   :\n{self.models[name]['cr']}\n"
        delimiter   = "#" * 100 + "\n\n"
        results     = model + params + train_score + test_score + cf_matrix + report + delimiter
        with open(filename, "a+", encoding = "utf-8") as file:
            file.write(results)
            file.close()
