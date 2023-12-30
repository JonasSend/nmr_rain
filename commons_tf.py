import commons
import pandas as pd
import math
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1_l2


def build_and_compile_keras_model(size: int, parameters: dict) -> tf.keras.Sequential:
    model = keras.Sequential()
    model.add(Dropout(input_shape=(size,), rate=parameters["dropout_in"]))
    for i in range(1, parameters["layers"] + 1):
        model.add(Dense(parameters["n" + str(i)], activation="relu",
                        kernel_regularizer=l1_l2(l1=parameters["l_1"], l2=parameters["l_2"])))
        model.add(Dropout(rate=parameters["dropout_hidden"]))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=pearson_correlation_loss,
                  optimizer=keras.optimizers.Adam(learning_rate=parameters["learning_rate"]))

    return model


def pearson_correlation_loss(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return -tfp.stats.correlation(x, y, sample_axis=0, event_axis=None)


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, df_X_validate: pd.DataFrame, y_validate: pd.Series, _era_validate: pd.Series,
                 stop: int, max_epochs: int, index: int, description: str) -> None:
        self.df_X_validate = df_X_validate
        self.y_validate = y_validate
        self.era_validate = _era_validate
        self.best_score = 0
        self.best_epoch = -1
        self.stop = stop
        self.max_epochs = max_epochs
        self.index = index
        self.description = description

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        prediction = pd.Series(self.model.predict(self.df_X_validate, verbose=0).flatten())
        correlation = commons.mean_grouped_spearman_correlation(prediction, self.y_validate, self.era_validate)

        if math.isnan(correlation):
            correlation = 0

        if correlation > self.best_score:
            self.best_score = correlation
            self.best_epoch = epoch
            self.model.save("./models/keras_" + self.description + "model_" + str(self.index) + ".h5")

        if (epoch >= (self.best_epoch + self.stop)) or (epoch >= (self.max_epochs - 1)):
            self.model.stop_training = True

        if epoch % round(self.stop/2) == 0:
            print(f"epoch: {epoch} . . . correlation: {correlation:.5f}")
