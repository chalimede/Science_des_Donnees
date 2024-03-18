# pylint: disable=line-too-long

""" params_dl module """

################################################################
## Copyright(C) 2024, Charles Theetten, <chalimede@proton.me> ##
################################################################

################################################################################

from    keras.layers            import Dense
from    keras.layers            import Dropout
from    keras.layers            import Input
from    keras.layers            import SimpleRNN
from    keras.layers            import LSTM
from    keras.models            import Sequential
from    keras.optimizers        import Adam
from    keras_tuner             import HyperModel

################################################################################

LAYERS = 23

################################################################################

class IxHyperModel(HyperModel):
    """ IxHyperModel class """

    def build(self, hp):
        model           = Sequential()
        learning_rate   = hp.Choice("learning_rate", values = [ 0.01, 0.005, 0.001 ])

        model.add(Input(shape = LAYERS))
        model.add(Dense(units = 1,
                        activation = "sigmoid"))

        model.compile(loss      = "binary_crossentropy",
                      metrics   = [ "accuracy" ],
                      optimizer = Adam(learning_rate = learning_rate))
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [ 64, 128, 365 ])
        return model.fit(*args, batch_size = batch_size, **kwargs)

################################################################################

class IdxHyperModel(HyperModel):
    """ IdxHyperModel class """

    def build(self, hp):
        model           = Sequential()
        learning_rate   = hp.Choice("learning_rate", values = [ 0.01, 0.005, 0.001 ])

        model.add(Input(shape = LAYERS))
        model.add(Dense(units       = hp.Int("units", min_value = 12, max_value = 48, step = 2),
                        activation  = hp.Choice("activation", [ "sigmoid" ])))
        model.add(Dense(units = 1,
                        activation = "sigmoid"))

        model.compile(loss      = "binary_crossentropy",
                      metrics   = [ "accuracy" ],
                      optimizer = Adam(learning_rate = learning_rate))
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [ 64, 128, 365 ])
        return model.fit(*args, batch_size = batch_size, **kwargs)

################################################################################

class IddxHyperModel(HyperModel):
    """ IddxHyperModel class """

    def build(self, hp):
        model           = Sequential()
        learning_rate   = hp.Choice("learning_rate", values = [ 0.01, 0.005, 0.001 ])

        model.add(Input(shape = LAYERS))
        model.add(Dense(units       = hp.Int("units", min_value = 12, max_value = 48, step = 2),
                        activation  = hp.Choice("activation", [ "sigmoid" ])))
        model.add(Dense(units       = hp.Int("units", min_value = 12, max_value = 48, step = 2),
                        activation  = hp.Choice("activation", [ "sigmoid" ])))
        model.add(Dense(units = 1,
                        activation = "sigmoid"))

        model.compile(loss      = "binary_crossentropy",
                      metrics   = [ "accuracy" ],
                      optimizer = Adam(learning_rate = learning_rate))
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [ 64, 128, 365 ])
        return model.fit(*args, batch_size = batch_size, **kwargs)

################################################################################

class IdOdxHyperModel(HyperModel):
    """ IdOdxHyperModel class """

    def build(self, hp):
        model           = Sequential()
        learning_rate   = hp.Choice("learning_rate", values = [ 0.01, 0.005, 0.001 ])

        model.add(Input(shape = LAYERS))
        model.add(Dense(units       = hp.Int("units", min_value = 12, max_value = 48, step = 2),
                        activation  = hp.Choice("activation", [ "sigmoid" ])))
        model.add(Dropout(rate      = hp.Float("rate", min_value = 0.25, max_value = 0.50)))
        model.add(Dense(units       = hp.Int("units", min_value = 12, max_value = 48, step = 2),
                        activation  = hp.Choice("activation", [ "sigmoid" ])))
        model.add(Dropout(rate      = hp.Float("rate", min_value = 0.125, max_value = 0.25)))
        model.add(Dense(units = 1,
                        activation = "sigmoid"))

        model.compile(loss      = "binary_crossentropy",
                      metrics   = [ "accuracy" ],
                      optimizer = Adam(learning_rate = learning_rate))
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [ 64, 128, 365 ])
        return model.fit(*args, batch_size = batch_size, **kwargs)

################################################################################

class SrnnHyperModel(HyperModel):
    """ SrnnHyperModel class """

    def build(self, hp):
        model           = Sequential()
        learning_rate   = hp.Choice("learning_rate", values = [ 0.01, 0.001 ])

        model.add(Input(shape = (LAYERS, 1)))
        model.add(SimpleRNN(units               = hp.Int("units", min_value = 36, max_value = 48, step = 2),
                            return_sequences    = True,
                            activation          = "tanh"))
        model.add(Dense(units       = hp.Int("units", min_value = 36, max_value = 48, step = 2),
                        activation  = hp.Choice("activation", [ "sigmoid" ])))
        model.add(Dense(units       = hp.Int("units", min_value = 36, max_value = 48, step = 2),
                        activation  = hp.Choice("activation", [ "sigmoid" ])))
        model.add(Dense(units = 1,
                        activation = "sigmoid"))

        model.compile(loss      = "binary_crossentropy",
                      metrics   = [ "accuracy" ],
                      optimizer = Adam(learning_rate = learning_rate))
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [ 128, 365 ])
        return model.fit(*args, batch_size = batch_size, **kwargs)

################################################################################

class LSTMHyperModel(HyperModel):
    """ LSTMHyperModel class """

    def build(self, hp):
        model           = Sequential()
        learning_rate   = hp.Choice("learning_rate", values = [ 0.01, 0.001 ])

        model.add(Input(shape = (LAYERS, 1)))
        model.add(LSTM(units                    = hp.Int("units", min_value = 24, max_value = 48, step = 2),
                            return_sequences    = True,
                            activation          = "sigmoid"))
        model.add(Dense(units       = hp.Int("units", min_value = 24, max_value = 48, step = 2),
                        activation  = hp.Choice("activation", [ "sigmoid" ])))
        model.add(Dense(units = 1,
                        activation = "sigmoid"))

        model.compile(loss      = "binary_crossentropy",
                      metrics   = [ "accuracy" ],
                      optimizer = Adam(learning_rate = learning_rate))
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [ 128, 365 ])
        return model.fit(*args, batch_size = batch_size, **kwargs)

################################################################################
