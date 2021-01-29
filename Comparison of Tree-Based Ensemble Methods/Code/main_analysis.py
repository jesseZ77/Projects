from models import *
from my_util import *
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

ENSEMBLES = ['Random Forest', 'Extra Trees', 'Adaboost', 'Gradient Boosting', 'XGBoost']
TICKERS = ['^AXJO', '^GSPC', '^FTSE', '^N225']
PROBLEM_TYPES = ['classification', 'regression']
STACKERS = ['Simple', 'Network', 'SVM']


def main(test=False, run_ensemble_analysis=True, run_ml_tuning=True, run_stacker_analysis=True):
    """
    test, if True, only performs a demonstration of the code and does not run the entire analysis
    run_ensemble_analysis: if True, performs hyper-parameter and analysis for individual ensembles
    run_ml_tuning: if True, performs hyper-parameter tuning for stacking model meta-learners
    run_stacker_analysis: if True, performs analysis for stacking models
    """
    if test:
        ticker = '^AXJO'
        problem_type = 'classification'

        # example of hyper-parameter tuning for individual ensembles
        data = load_data(ticker, data_type=problem_type)
        base = ExtraTreesClassifier()
        param_grid = {'max_depth': [2, 4], 'max_features': [2]}
        ensemble = TreeBasedEnsemble(name=f"test", data=data, problem_type=problem_type, model=base)
        ensemble.find_best_model(param_grid, save=False)

        # example of ensemble analysis
        ensemble.run_analysis(n_reps=1)
        results = ResultsWriter(ensemble)
        results.write_raw_results()

        # example of meta-learner tuning
        X_train_stack, X_test_stack = stack_ensemble_predictions(ticker, problem_type)
        np.savetxt(f"data/{ticker}/X_train_stack_{problem_type}.txt", X_train_stack)
        np.savetxt(f"data/{ticker}/X_test_stack_{problem_type}.txt", X_test_stack)

        data = load_stack_data(ticker, problem_type)
        param_grid = {'C': [0.01, 1], 'gamma': ['auto']}
        svm = MySVM(f"Meta learner test", data, problem_type)
        svm.find_best_model(param_grid)

        # example of stacker model analysis using the svm tuned previously
        data = load_data(ticker, data_type=problem_type)
        ensemble_list = [TreeBasedEnsemble(name=f"{algo} {problem_type} {ticker}",
                                           problem_type=problem_type,
                                           data=data, load_from_file=True) for algo in ENSEMBLES]
        stacker = Stacker(f'Stacker test', data, problem_type, ensemble_list, svm)
        stacker.run_analysis(1)
        results = ResultsWriter(stacker)
        results.write_raw_results()

        return 0

    if run_ensemble_analysis:
        print("Performing ensemble parameter tuning and analysis")
        for problem_type in PROBLEM_TYPES:
            for ticker in TICKERS:
                print(f'Running: {problem_type} {ticker}')
                ensemble_analysis(ticker, problem_type)
                print(f'Done: {problem_type} {ticker}')

    if run_ml_tuning:
        print("Performing meta-learner parameter tuning")
        stack_all_ensemble_predictions()
        for problem_type in PROBLEM_TYPES:
            for ticker in TICKERS:
                print(f'Running: {problem_type} {ticker}')
                # train_network_meta_learner(ticker, problem_type)
                train_svm_meta_learner(ticker, problem_type)
                print(f'Done: {problem_type} {ticker}')

    if run_stacker_analysis:
        print("Performing stacker analysis")
        for problem_type in PROBLEM_TYPES:
            for ticker in TICKERS:
                for ml_type in STACKERS:
                    print(f'Running: {problem_type} {ticker} {ml_type}')
                    stacker_analysis(ticker, problem_type, ml_type)
                    print(f'Done: {problem_type} {ticker} {ml_type}')


def ensemble_analysis(ticker, problem_type):
    """
    Performs parameter tuning for each ensemble, does 10 fits of the model and then writes results to file
    """
    data = load_data(ticker, data_type=problem_type)
    X_train, y_train, X_test, y_test = data
    if problem_type == 'classification':
        classifiers = {'Random Forest': RandomForestClassifier(),
                       'Extra Trees': ExtraTreesClassifier(),
                       'Adaboost': AdaBoostClassifier(),
                       'Gradient Boosting': GradientBoostingClassifier(),
                       'XGBoost': XGBClassifier()
                       }
    else:
        classifiers = {'Random Forest': RandomForestRegressor(),
                       'Extra Trees': ExtraTreesRegressor(),
                       'Adaboost': AdaBoostRegressor(),
                       'Gradient Boosting': GradientBoostingRegressor(),
                       'XGBoost': XGBRegressor()
                       }

    param_grids = {'Random Forest': {'max_depth': [2, 4, 8], 'max_features': [2, 4, 8]},
                   'Extra Trees': {'max_depth': [2, 4, 8], 'max_features': [2, 4, 8]},
                   'Adaboost': {'learning_rate': [0.01, 0.1, 1.0, 5.0]},
                   'Gradient Boosting':
                       {'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [2, 4, 8], 'max_features': [2, 4, 8]},
                   'XGBoost': {'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [2, 4, 8]}
                   }

    for algo in ENSEMBLES:
        print(f'Ensemble method: {algo}')
        base = classifiers[algo]
        param_grid = param_grids[algo]
        ensemble = TreeBasedEnsemble(name=f"{algo} {problem_type} {ticker}", data=data,
                                     problem_type=problem_type, model=base)
        search = ensemble.find_best_model(param_grid, save=False)
        best_model = search.best_estimator_

        ensemble.run_analysis(n_reps=10, save=True)
        results = ResultsWriter(ensemble)
        results.summarise()
        results.write_summary()
        results.write_raw_results()


def stack_all_ensemble_predictions():
    """
    Stacks predictions from individual ensembles and writes them to file
    Predictions are to be used for meta-learner hyper-parameter tuning
    """
    for problem_type in PROBLEM_TYPES:
        for ticker in TICKERS:
            X_train_stack, X_test_stack = stack_ensemble_predictions(ticker, problem_type)
            np.savetxt(f"data/{ticker}/X_train_stack_{problem_type}.txt", X_train_stack)
            np.savetxt(f"data/{ticker}/X_test_stack_{problem_type}.txt", X_test_stack)


def train_network_meta_learner(ticker, problem_type):
    """
    Performs parameter tuning for the neural network meta-learner
    """
    data = load_stack_data(ticker, problem_type)
    network = MyNetwork(f"Network {problem_type} {ticker}", data, problem_type)
    network.find_best_model([5, 10, 20], [1, 2, 3], [0.0, 0.1, 0.2, 0.3], save=True)


def train_svm_meta_learner(ticker, problem_type):
    """
    Performs parameter tuning for the SVM meta-learner
    """
    data = load_stack_data(ticker, problem_type)
    param_grid = {'C': [0.01, 1, 10, 100], 'gamma': ['auto']}
    svm = MySVM(f"SVM {problem_type} {ticker}", data, problem_type)
    svm.find_best_model(param_grid, save=True)


def stacker_analysis(ticker, problem_type, ml_type):
    """
    Does 10 fits for each stacking model, writing results to file
    """
    data = load_data(ticker, data_type=problem_type)
    X_train, y_train, X_test, y_test = data
    ensemble_list = [TreeBasedEnsemble(name=f"{algo} {problem_type} {ticker}", problem_type=problem_type,
                                       data=data, load_from_file=True) for algo in ENSEMBLES]

    X_train_stack = np.hstack([ensemble.predict(X_train) for ensemble in ensemble_list])
    X_test_stack = np.hstack([ensemble.predict(X_test) for ensemble in ensemble_list])

    data_stack = (X_train_stack, y_train, X_test_stack, y_test)

    if ml_type == "Network":
        ml = MyNetwork(f'Network {problem_type} {ticker}', data_stack, problem_type, load_from_file=True)
    elif ml_type == "SVM":
        ml = MySVM(f'SVM {problem_type} {ticker}', data_stack, problem_type, load_from_file=True)
    else:
        ml = SimpleMetalearner(f"Simple {problem_type} {ticker}", data_stack, problem_type)

    stacker = Stacker(f'Stacker {ml_type} {problem_type} {ticker}', data, problem_type, ensemble_list, ml)
    stacker.run_analysis(10)

    results = ResultsWriter(stacker)
    results.summarise()
    results.write_summary()
    results.write_raw_results()


if __name__ == "__main__":
    main(test=True)
