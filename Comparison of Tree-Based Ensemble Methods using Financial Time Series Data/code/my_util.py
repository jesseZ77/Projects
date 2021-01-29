import numpy as np
import pandas as pd
from scipy.stats import t
from models import TreeBasedEnsemble
import re

ENSEMBLES = ['Random Forest', 'Extra Trees', 'Adaboost', 'Gradient Boosting', 'XGBoost']
ENSEMBLES_SHORT = ['RF', 'ET', 'ADA', 'GB', 'XGB']
TICKERS = ['^AXJO', '^GSPC', '^FTSE', '^N225']
TICKERS_PRINT = [ticker[1:] for ticker in TICKERS]


def load_data(ticker, data_type='classification'):
    X_train = np.loadtxt(f"data/{ticker}/X_train.txt")
    X_test = np.loadtxt(f"data/{ticker}/X_test.txt")
    
    if data_type == 'classification':
        y_train = np.loadtxt(f"data/{ticker}/y_train_classification.txt")
        y_test = np.loadtxt(f"data/{ticker}/y_test_classification.txt")
        
    elif data_type == 'regression':
        y_train = np.loadtxt(f"data/{ticker}/y_train_regression.txt")
        y_test = np.loadtxt(f"data/{ticker}/y_test_regression.txt")

    elif data_type == 'multistep':
        y_train = np.loadtxt(f"data/{ticker}/y_train_multistep.txt")
        y_test = np.loadtxt(f"data/{ticker}/y_test_multistep.txt")

    else:
        raise Exception("Not a valid data type")

    return X_train, y_train, X_test, y_test


class ResultsWriter:
    def __init__(self, ensemble):
        self.results = ensemble.results
        self.n = len(ensemble.results[list(ensemble.results.keys())[0]])
        self.name = ensemble.name
        self.results_summary = None

    def summarise(self):
        self.results_summary = pd.DataFrame({
            "Mean": self.results.mean(),
            "Std. dev": self.results.std(),
            "CI: lower": self.results.mean() - t.ppf(0.975, self.n - 1) * self.results.std() / np.sqrt(self.n),
            "CI: upper": self.results.mean() + t.ppf(0.975, self.n - 1) * self.results.std() / np.sqrt(self.n)
        })

    def write_summary(self):
        print('Writing results summary')
        self.results_summary.to_pickle(f'results/results_summary_{self.name}.pkl')

        latex_table = (self.results_summary
                       .applymap(lambda x: str(int(x)) if abs(x - int(x)) < 1e-6 else str(round(x, 4)))
                       .to_latex(caption=self.name, bold_rows=True))

        latex_table = re.sub("\\\\begin{table}", "\\\\begin{table}[h]", latex_table)
        text_file = open(f'results/latex tables/results_summary_{self.name}.txt', "w")
        text_file.write(latex_table)
        text_file.close()

    def write_raw_results(self):
        print('Writing raw results')
        self.results.to_pickle(f'results/results_{self.name}.pkl')


def stack_ensemble_predictions(ticker, problem_type):
    data = load_data(ticker, data_type=problem_type)
    X_train, y_train, X_test, y_test = data
    ensemble_list = [TreeBasedEnsemble(name=f"{algo} {problem_type} {ticker}", problem_type=problem_type,
                                       data=data, load_from_file=True) for algo in ENSEMBLES]

    X_train_stack = np.hstack([ensemble.predict(X_train) for ensemble in ensemble_list])
    X_test_stack = np.hstack([ensemble.predict(X_test) for ensemble in ensemble_list])

    return X_train_stack, X_test_stack


def load_stack_data(ticker, problem_type):
    X_train_stack = np.loadtxt(f"data/{ticker}/X_train_stack_{problem_type}.txt")
    X_test_stack = np.loadtxt(f"data/{ticker}/X_test_stack_{problem_type}.txt")
    y_train = np.loadtxt(f"data/{ticker}/y_train_{problem_type}.txt")
    y_test = np.loadtxt(f"data/{ticker}/y_test_{problem_type}.txt")

    return X_train_stack, y_train, X_test_stack, y_test