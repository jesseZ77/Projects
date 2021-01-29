import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam

from time import time
import re

import matplotlib.pyplot as plt

from scipy.stats import t


class ClassificationModel:
    def __init__(self, train_data, val_data, test_data, caption, n_reps=10, optim="sgd", lr=0.01, mom=0.0,
                 n_hidden_layers=1, n_hidden_neurons=10):
        self.X_train = train_data[0]
        self.y_train = train_data[1]
        self.X_val = val_data[0]
        self.y_val = val_data[1]
        self.X_test = test_data[0]
        self.y_test = test_data[1]
        self.caption = caption
        self.n_reps = n_reps
        self.optim = optim
        self.lr = lr
        self.mom = mom
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.build_and_compile_model()

    def build_and_compile_model(self):
        i = Input(shape=self.X_train[0].shape)
        x = Dense(self.n_hidden_neurons, activation="relu")(i)
        for _ in range(self.n_hidden_layers - 1):
            x = Dense(self.n_hidden_neurons, activation="relu")(x)
        o = Dense(1, activation='sigmoid')(x)
        model = Model(i, o)

        if self.optim == "sgd":
            optimizer = SGD(learning_rate=self.lr, momentum=self.mom)
        elif self.optim == "adam":
            optimizer = Adam()

        model.compile(optimizer=optimizer,
                      loss='BinaryCrossentropy',
                      metrics=['accuracy'])
        self.model = model

    def evaluate(self, write=True):
        results = pd.DataFrame(
            columns=["Loss: train", "Loss: val", "Loss: test",
                     "Acc: train", "Acc: val", "Acc: test",
                     "Num epochs", "Time (s)"])

        for i in range(self.n_reps):
            print(f"{self.caption}: starting run {i + 1} out of {self.n_reps}")
            early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
            self.build_and_compile_model()

            start = time()
            r = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                               epochs=300, verbose=False, callbacks=[early_stopping_cb])
            time_taken = time() - start

            loss_train, acc_train = self.model.evaluate(self.X_train, self.y_train, verbose=False)
            loss_validation, acc_validation = self.model.evaluate(self.X_val, self.y_val, verbose=False)
            loss_test, acc_test = self.model.evaluate(self.X_test, self.y_test, verbose=False)

            row = pd.DataFrame({
                "Loss: train": [loss_train],
                "Loss: val": [loss_validation],
                "Loss: test": [loss_test],
                "Acc: train": [acc_train],
                "Acc: val": [acc_validation],
                "Acc: test": [acc_test],
                "Num epochs": [len(r.history["loss"])],
                "Time (s)": time_taken
            })
            results = results.append(row, ignore_index=True)

        results_summary = pd.DataFrame({
            "Mean": results.mean(),
            "Std. dev": results.std(),
            "CI: lower": results.mean() - t.ppf(0.975, self.n_reps - 1) * results.std() / np.sqrt(self.n_reps),
            "CI: upper": results.mean() + t.ppf(0.975, self.n_reps - 1) * results.std() / np.sqrt(self.n_reps)
        })

        results_summary[results_summary < 0] = 0

        if write:
            self.write_data(results_summary, r)

        return r, results, results_summary

    def write_data(self, results_summary, r):
        self._write_table(results_summary)
        self._write_image(r)

    def _write_table(self, results_summary):
        table = (results_summary
                 .applymap(lambda x: str(int(x)) if abs(x - int(x)) < 1e-6 else str(round(x, 4)))
                 .to_latex(caption=self.caption, bold_rows=True))
        table = re.sub("\\\\begin{table}", "\\\\begin{table}[H]", table)
        text_file = open(f'Tables/{self.caption}.txt', "w")
        text_file.write(table)
        text_file.close()

    def _write_image(self, r):
        fig = plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.title(f"Loss per iteration - {self.caption}")
        plt.plot(r.history['loss'], label='Loss')
        plt.plot(r.history['val_loss'], label='Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title(f"Accuracy per iteration - {self.caption}")
        plt.plot(r.history['accuracy'], label='Accuracy')
        plt.plot(r.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        fig.savefig(f"Images/{self.caption}.pdf")
        plt.close()