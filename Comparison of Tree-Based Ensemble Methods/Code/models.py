import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from joblib import dump, load


from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression


class TreeBasedEnsemble:
    def __init__(self, name, data, problem_type='classification',
                 model=None, load_from_file=False):
        self.name = name
        self._unpack_data(data)
        self.problem_type = problem_type

        if load_from_file:
            self._load_model(self.name)
        else:
            self.model = model
        self.model.n_estimators = 400

        self.results = None

    def find_best_model(self, param_grid, save=False):
        search = GridSearchCV(self.model, param_grid=param_grid, cv=10, verbose=1, n_jobs=-1)
        search.fit(self.X_train, self.y_train)

        print(search.best_params_)
        self.model = search.best_estimator_

        if save:
            print("saving model")
            dump(self.model, f'models/{self.name}.joblib')

        return search

    def run_analysis(self, n_reps=10, save=False):
        if self.problem_type == 'classification':
            results = {'accuracy: train': [],
                       'accuracy: test': []}
        else:
            results = {'rmse: train': [],
                       'rmse: test': []}

        for i in range(n_reps):
            print(f"starting rep {i + 1} / {n_reps}")
            self.train(self.X_train, self.y_train)

            if self.problem_type == 'classification':
                y_pred_train = np.argmax(self.predict(self.X_train), axis=1)
                y_pred_test = np.argmax(self.predict(self.X_test), axis=1)
                results['accuracy: train'].append(accuracy_score(self.y_train, y_pred_train))
                results['accuracy: test'].append(accuracy_score(self.y_test, y_pred_test))

            else:
                y_pred_train = self.predict(self.X_train)
                y_pred_test = self.predict(self.X_test)
                results['rmse: train'].append(100 * mean_squared_error(self.y_train, y_pred_train) ** 0.5)
                results['rmse: test'].append(100 * mean_squared_error(self.y_test, y_pred_test) ** 0.5)

        self.results = pd.DataFrame(results)

        if save:
            print("saving model")
            dump(self.model, f'models/{self.name}.joblib')

    def predict(self, X):
        if self.problem_type == "classification":
            return self.model.predict_proba(X)
        elif self.problem_type == "regression":
            return self.model.predict(X).reshape(-1, 1)
        else:
            return self.model.predict(X)

    def train(self, X, y):
        self.model.fit(X, y)

    def _unpack_data(self, data):
        self.X_train = data[0]
        self.y_train = data[1]
        self.X_test = data[2]
        self.y_test = data[3]

    def _load_model(self, name):
        self.model = load(f'models/{name}.joblib')


class MyNetwork:
    def __init__(self, name, data, problem_type, load_from_file=False):
        self.name = name
        self.problem_type = problem_type
        self._unpack_data(data)

        if problem_type == 'classification':
            self.output_activation = "softmax"
            self.n_output_neurons = len(np.unique(self.y_train))
            self.loss = "sparse_categorical_crossentropy"
        else:
            self.output_activation = 'linear'
            self.n_output_neurons = 1
            self.loss = 'mse'

        if load_from_file:
            self._load_model()
        else:
            self.n_hidden_neurons = 10
            self.n_hidden_layers = 1
            self.dropout = 0.20
            self._build_and_compile_model()

    def find_best_model(self, n_hidden_neurons, n_hidden_layers, dropout_pct, save=False):
        best_loss = np.inf
        best_hidden_neurons = self.n_hidden_neurons
        best_hidden_layers = self.n_hidden_layers
        best_dropout = self.dropout

        X_split_list = np.split(self.X_train, [int(x) for x in len(self.X_train) / 10 * np.arange(1, 10)])
        y_split_list = np.split(self.y_train, [int(y) for y in len(self.y_train) / 10 * np.arange(1, 10)])

        for neurons in n_hidden_neurons:
            print("Finding best number of hidden neurons")
            self.n_hidden_neurons = neurons
            losses = []
            for i in range(10):
                print(f'Neurons: {neurons}, fold: {i + 1}')
                X_train_tmp, y_train_tmp, X_test_tmp, y_test_tmp = self._cv_split(X_split_list, y_split_list, i)
                _ = self.train(X_train_tmp, y_train_tmp, build_new=True)
                losses.append(self.model.evaluate(X_test_tmp, y_test_tmp, verbose=0))

            if np.mean(losses) < best_loss:
                print("found better model")
                best_loss = np.mean(losses)
                best_hidden_neurons = neurons

        self.n_hidden_neurons = best_hidden_neurons
        # self._write_results_live()
        print("finished neurons investigation")

        for layers in n_hidden_layers:
            print("Finding best number of hidden layers")
            self.n_hidden_layers = layers
            losses = []
            for i in range(10):
                print(f'Layers: {layers}, fold: {i + 1}')
                X_train_tmp, y_train_tmp, X_test_tmp, y_test_tmp = self._cv_split(X_split_list, y_split_list, i)
                _ = self.train(X_train_tmp, y_train_tmp, build_new=True)
                losses.append(self.model.evaluate(X_test_tmp, y_test_tmp, verbose=0))

            if np.mean(losses) < best_loss:
                print("found better model")
                best_loss = np.mean(losses)
                best_hidden_layers = layers
        self.n_hidden_layers = best_hidden_layers
        # self._write_results_live()
        print("finished layers investigation")

        for dropout in dropout_pct:
            print("Finding best dropout")
            self.dropout = dropout
            losses = []
            for i in range(10):
                print(f'Dropout: {dropout}, fold: {i + 1}')
                X_train_tmp, y_train_tmp, X_test_tmp, y_test_tmp = self._cv_split(X_split_list, y_split_list, i)
                _ = self.train(X_train_tmp, y_train_tmp, build_new=True)
                losses.append(self.model.evaluate(X_test_tmp, y_test_tmp, verbose=0))

            if np.mean(losses) < best_loss:
                print("found better model")
                best_loss = np.mean(losses)
                best_dropout = dropout
        self.dropout = best_dropout
        # self._write_results_live()
        print("finished dropout investigation")

        print(f'Best parameters - '
              f'Neurons: {best_hidden_neurons}, '
              f'Layers: {best_hidden_layers}, '
              f'Dropout: {best_dropout}')
        if save:
            np.savetxt(f'models/{self.name}.txt', np.array([best_hidden_neurons, best_hidden_layers, best_dropout]))

        return best_hidden_neurons, best_hidden_layers, dropout

    def train(self, X, y, build_new=True):
        if build_new:
            self._build_and_compile_model()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

        early_stopping_cb = EarlyStopping(patience=25, restore_best_weights=True)
        r = self.model.fit(X_train, y_train,
                           validation_data=(X_val, y_val),
                           epochs=300, verbose=False, callbacks=[early_stopping_cb])
        print("finished training model")
        return r

    def predict(self, X):
        return self.model.predict(X)

    def _load_model(self):
        params = np.loadtxt(f'models/{self.name}.txt')
        self.n_hidden_neurons = int(params[0])
        self.n_hidden_layers = int(params[1])
        self.dropout = params[2]
        self._build_and_compile_model()

    def _build_and_compile_model(self):
        i = Input(shape=self.X_train[0].shape)
        x = Dense(self.n_hidden_neurons, activation="relu")(i)
        x = Dropout(rate=self.dropout)(x)
        for _ in range(self.n_hidden_layers - 1):
            x = Dense(self.n_hidden_neurons, activation="relu")(x)
            x = Dropout(rate=self.dropout)(x)
        o = Dense(self.n_output_neurons, activation=self.output_activation)(x)
        model = Model(i, o)
        self.model = model
        self.model.compile(optimizer="adam", loss=self.loss)

    def _unpack_data(self, data):
        self.X_train = data[0]
        self.y_train = data[1]
        self.X_test = data[2]
        self.y_test = data[3]

    def _cv_split(self, X_split_list, y_split_list, n):
        X_train = np.vstack([X_split_list[i] for i in range(10) if i != n])
        y_train = np.vstack([y_split_list[i].reshape(-1, 1) for i in range(10) if i != n])
        X_test = X_split_list[n]
        y_test = y_split_list[n].reshape(-1, 1)

        return X_train, y_train, X_test, y_test

    def _write_results_live(self):
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        if self.problem_type == 'classification':
            y_pred_train = np.argmax(y_pred_train, axis=1)
            y_pred_test = np.argmax(y_pred_test, axis=1)
            row = pd.DataFrame({
                'train accuracy': accuracy_score(self.y_train, y_pred_train),
                'test accuracy': accuracy_score(self.y_test, y_pred_test),
            }, index=[f'Network: {self.n_hidden_neurons}-{self.n_hidden_layers}-{self.dropout}'])
        else:
            row = pd.DataFrame({
                'train rmse': 100 * mean_squared_error(self.y_train, y_pred_train) ** 0.5,
                'test rmse': 100 * mean_squared_error(self.y_test, y_pred_test) ** 0.5,
            }, index=[f'Network: {self.n_hidden_neurons}-{self.n_hidden_layers}-{self.dropout}'])
        print("writing results")
        tmp_results = pd.read_pickle(f"live_results_network_{self.problem_type}_{self.name[-4:]}.pkl")
        tmp_results = tmp_results.append(row)
        tmp_results.to_pickle(f"live_results_network_{self.problem_type}_{self.name[-4:]}.pkl")


class MySVM:
    def __init__(self, name, data, problem_type, load_from_file=False):
        self.name = name
        self._unpack_data(data)
        self.problem_type = problem_type

        if load_from_file:
            self._load_model(self.name)
        else:
            if problem_type == 'classification':
                self.model = SVC(probability=True)
            else:
                self.model = SVR()

    def find_best_model(self, param_grid, save=False):
        search = GridSearchCV(self.model, param_grid=param_grid, cv=10, verbose=1, n_jobs=-1)
        search.fit(self.X_train, self.y_train)

        print(search.best_params_)
        self.model = search.best_estimator_

        if save:
            print("saving model")
            dump(self.model, f'models/{self.name}.joblib')

        return search

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.problem_type == "classification":
            return self.model.predict_proba(X)
        elif self.problem_type == "regression":
            return self.model.predict(X).reshape(-1, 1)

    def _unpack_data(self, data):
        self.X_train = data[0]
        self.y_train = data[1]
        self.X_test = data[2]
        self.y_test = data[3]

    def _load_model(self, name):
        self.model = load(f'models/{name}.joblib')


class SimpleMetalearner:
    def __init__(self, name, data, problem_type, load_from_file=False):
        self.name = name
        self._unpack_data(data)
        self.problem_type = problem_type

        if load_from_file:
            self._load_model(self.name)
        else:
            if problem_type == 'classification':
                self.model = LogisticRegression(multi_class='auto', solver='lbfgs')
            else:
                self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.problem_type == "classification":
            return self.model.predict_proba(X)
        elif self.problem_type == "regression":
            return self.model.predict(X).reshape(-1, 1)

    def _unpack_data(self, data):
        self.X_train = data[0]
        self.y_train = data[1]
        self.X_test = data[2]
        self.y_test = data[3]

    def _load_model(self, name):
        self.model = load(f'models/{name}.joblib')


class Stacker:
    def __init__(self, name, data, problem_type, ensemble_list, meta_learner):
        self.name = name
        self._unpack_data(data)
        self.problem_type = problem_type
        self.ensemble_list = ensemble_list
        self.meta_learner = meta_learner
        self.results = None

    def train(self, X, y):
        i = 1
        for ensemble in self.ensemble_list:
            print(f"training ensembles: {i} / {len(self.ensemble_list)}")
            ensemble.train(X, y)
            i += 1

        print("training meta learner")
        X_train_stack = np.hstack([ensemble.predict(X) for ensemble in self.ensemble_list])
        self.meta_learner.train(X_train_stack, y)

    def predict(self, X):
        X_stack = np.hstack([ensemble.predict(X) for ensemble in self.ensemble_list])
        y_pred = self.meta_learner.predict(X_stack)
        return y_pred

    def run_analysis(self, n_reps=10):
        if self.problem_type == 'classification':
            results = {'accuracy: train': [],
                       'accuracy: test': []}
        else:
            results = {'rmse: train': [],
                       'rmse: test': []}

        for i in range(n_reps):
            print(f"starting rep {i + 1} / {n_reps}")
            self.train(self.X_train, self.y_train)

            if self.problem_type == 'classification':
                y_pred_train = np.argmax(self.predict(self.X_train), axis=1)
                y_pred_test = np.argmax(self.predict(self.X_test), axis=1)
                results['accuracy: train'].append(accuracy_score(self.y_train, y_pred_train))
                results['accuracy: test'].append(accuracy_score(self.y_test, y_pred_test))

            else:
                y_pred_train = self.predict(self.X_train)
                y_pred_test = self.predict(self.X_test)
                results['rmse: train'].append(100 * mean_squared_error(self.y_train, y_pred_train) ** 0.5)
                results['rmse: test'].append(100 * mean_squared_error(self.y_test, y_pred_test) ** 0.5)

        self.results = pd.DataFrame(results)

    def _unpack_data(self, data):
        self.X_train = data[0]
        self.y_train = data[1]
        self.X_test = data[2]
        self.y_test = data[3]
