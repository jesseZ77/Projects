import matplotlib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from my_util import *
from models import *
import pandas as pd
import numpy as np
from joblib import load

ENSEMBLES = ['Random Forest', 'Extra Trees', 'Adaboost', 'Gradient Boosting', 'XGBoost']
ENSEMBLES_SHORT = ['RF', 'ET', 'ADA', 'GB', 'XGB']
ENSEMBLES_SHORT_dict = dict(zip(ENSEMBLES, ENSEMBLES_SHORT))
STACKERS = ["Simple", "Network", "SVM"]
TICKERS = ['^AXJO', '^GSPC', '^FTSE', '^N225']
TICKERS_PRINT = [ticker[1:] for ticker in TICKERS]
problem_types = ['classification', 'regression']
matplotlib.rcParams.update({'font.size': 18})


def save_image(fig, file_name=None):
    fig.savefig(f"Report/Images/{file_name}.pdf")


def save_as_latex(df, caption=None, file_name=None, save=True):
    if file_name is None:
        file_name = caption
    table = (df
             .applymap(lambda x: str(int(x)) if abs(x - int(x)) < 1e-6 else str(round(x, 4)))
             .to_latex(caption=caption, bold_rows=True))
    table = re.sub("\\\\begin{table}", "\\\\begin{table}[H]", table)
    if save:
        text_file = open(f'Report/Tables/{file_name}.txt', "w")
        text_file.write(table)
        text_file.close()

    return table


def summary_stats():
    data_list = [pd.read_pickle(f"data/{ticker}/data.pkl") for ticker in TICKERS]
    sum_stats_list = [pd.read_pickle(f"data/{ticker}/data.pkl")[['returns']]
                          .assign(returns=lambda x: 100 * x.returns)
                          .describe().rename(columns={'returns': ticker[1:]})[1:] for ticker in TICKERS]

    n_obs = np.array([len(data) for data in data_list])
    n_test = np.floor(n_obs * 0.3)
    n_train = n_obs - n_test

    df_counts = pd.DataFrame({
        'count': n_obs,
        'train count': n_train,
        'test count': n_test}, index=TICKERS_PRINT).T

    df_sum_stats = pd.concat(sum_stats_list, axis=1)

    df_sum_stats = pd.concat([df_counts, df_sum_stats], axis=0)

    return df_sum_stats


def price_return_plots(ticker):
    data = pd.read_pickle(f"data/{ticker}/data.pkl")
    data['returns'] = 100 * data['returns']
    data['test'] = 0
    data.loc[int(len(data) * (1 - 0.3)):, 'test'] = 1

    fig1 = plt.figure(figsize=(8, 6))
    plt.title(f'{ticker[1:]}: Price', fontsize=22)
    data[lambda x: x.test == 0].set_index("date")['price'].plot(label='train')
    data[lambda x: x.test == 1].set_index("date")['price'].plot(label='test')
    plt.ylabel('index', fontsize=18)
    plt.xlabel('date', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.close()

    fig2 = plt.figure(figsize=(8, 6))
    plt.title(f'{ticker[1:]}: Returns', fontsize=22)

    data[lambda x: x.test == 0]['returns'].plot.hist(bins=int(np.sqrt(len(data[lambda x: x.test == 0]))),
                                                     density=True, label='train', alpha=0.4)
    data[lambda x: x.test == 1]['returns'].plot.hist(bins=int(np.sqrt(len(data[lambda x: x.test == 0]))),
                                                     density=True, label='test', alpha=0.4)
    plt.ylabel('density', fontsize=18)
    plt.xlabel('return (%)', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.close()

    return fig1, fig2


def cutoffs_table():
    cutoffs = np.loadtxt("data/cutoff.txt") * 100
    df_cutoffs = pd.DataFrame(cutoffs, columns=['Down Cut-off', 'Up Cut-off'], index=TICKERS_PRINT)

    return df_cutoffs


def class_distributions():
    rows = []
    row_labels = []
    for ticker in TICKERS:
        for dataset in ["train", "test"]:
            _, counts = np.unique(np.loadtxt(f"data/{ticker}/y_{dataset}_classification.txt"), return_counts=True)
            rows.append(counts)
            row_labels.append(ticker[1:] + ": " + dataset)
    df_out = pd.DataFrame(rows, index=row_labels, columns=['Down', 'Unchanged', 'Up'])
    return df_out


def regression_data_tables():
    df_train = pd.concat(
        [pd.DataFrame(np.loadtxt(f"data/{ticker}/y_train_regression.txt") * 100).describe() for ticker in TICKERS],
        axis=1)

    df_train.columns = TICKERS_PRINT

    df_test = pd.concat(
        [pd.DataFrame(np.loadtxt(f"data/{ticker}/y_test_regression.txt") * 100).describe() for ticker in TICKERS],
        axis=1)

    df_test.columns = TICKERS_PRINT

    return df_train, df_test


def regression_target_plots(ticker):
    train = np.loadtxt(f"data/{ticker}/y_train_regression.txt") * 100
    test = np.loadtxt(f"data/{ticker}/y_test_regression.txt") * 100

    fig = plt.figure(figsize=(8, 6))
    plt.title(f'{ticker[1:]}: 5-day Returns', fontsize=22)

    plt.hist(train, bins=int(np.sqrt(len(train))), density=True, label="train", alpha=0.4)
    plt.hist(test, bins=int(np.sqrt(len(test))), density=True, label="test", alpha=0.4)

    plt.ylabel('density', fontsize=18)
    plt.xlabel('return (%)', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.close()

    return fig


def results_summary_table(ticker, problem_type):
    df_list = []
    metric = 'accuracy' if problem_type == "classification" else 'rmse'

    for algo in ENSEMBLES:
        df_tmp = pd.read_pickle(f"results/results_summary_{algo} {problem_type} {ticker}.pkl")
        df_tmp.rename({f'{metric}: train': f'train: {algo}',
                       f'{metric}: test': f'test: {algo}'}, inplace=True)
        df_list.append(df_tmp)

    for algo in STACKERS:
        df_tmp = pd.read_pickle(f"results/results_summary_Stacker {algo} {problem_type} {ticker}.pkl")
        df_tmp.rename({f'{metric}: train': f'train: {algo}',
                       f'{metric}: test': f'test: {algo}'}, inplace=True)
        df_list.append(df_tmp)
    df_concat = pd.concat(df_list, axis=0)

    df_concat_train = df_concat.loc[[row for row in df_concat.index if 'train' in row], :]
    df_concat_test = df_concat.loc[[row for row in df_concat.index if 'test' in row], :]
    simple_train, simple_test = simple_model_metrics(ticker, problem_type)

    df_concat_train.rename(dict(zip(['train: ' + ensemble for ensemble in ENSEMBLES], ENSEMBLES)), inplace=True)
    df_concat_train.rename(dict(zip(['train: ' + stacker for stacker in STACKERS],
                                    [stacker + ' Stacker' for stacker in STACKERS])), inplace=True)
    df_concat_train.rename(dict(zip(ENSEMBLES, ENSEMBLES_SHORT)), inplace=True)
    df_concat_train = df_concat_train.append(simple_train)

    df_concat_test.rename(dict(zip(['test: ' + ensemble for ensemble in ENSEMBLES], ENSEMBLES)), inplace=True)
    df_concat_test.rename(dict(zip(['test: ' + stacker for stacker in STACKERS],
                                   [stacker + ' Stacker' for stacker in STACKERS])), inplace=True)
    df_concat_test.rename(dict(zip(ENSEMBLES, ENSEMBLES_SHORT)), inplace=True)
    df_concat_test = df_concat_test.append(simple_test)

    ascending = False if problem_type == 'classification' else True
    df_concat_train['rank'] = df_concat_train['Mean'].rank(ascending=ascending).astype("int")
    df_concat_test['rank'] = df_concat_test['Mean'].rank(ascending=ascending).astype("int")

    return df_concat_train, df_concat_test


def summarise_ranks(problem_type):
    train_ranks = []
    test_ranks = []

    for ticker in TICKERS:
        df_concat_train, df_concat_test = results_summary_table(ticker, problem_type)
        train_ranks.append(df_concat_train['rank'].values)
        test_ranks.append(df_concat_test['rank'].values)

    train_ranks = np.array(train_ranks).T
    test_ranks = np.array(test_ranks).T

    rank_summary_train = pd.DataFrame(train_ranks, columns=TICKERS_PRINT,
                                      index=ENSEMBLES_SHORT +
                                            [stacker + ' Stacker'  for stacker in STACKERS] +
                                            ['Simple'])
    rank_summary_train['average'] = rank_summary_train.mean(axis=1)

    rank_summary_test = pd.DataFrame(test_ranks, columns=TICKERS_PRINT,
                                     index=ENSEMBLES_SHORT +
                                           [stacker + ' Stacker' for stacker in STACKERS] +
                                           ['Simple'])
    rank_summary_test['average'] = rank_summary_test.mean(axis=1)

    rank_summary_train = rank_summary_train.round(2)

    return rank_summary_train, rank_summary_test


def simple_model_metrics(ticker, problem_type):
    data = load_data(ticker, problem_type)
    X_train, y_train, X_test, y_test = data

    if problem_type == "classification":
        reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        reg.fit(X_train, y_train)
        train_score, test_score = (accuracy_score(y_train, reg.predict(X_train)),
                                   accuracy_score(y_test, reg.predict(X_test)))
    else:
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        train_score, test_score = (100 * mean_squared_error(y_train, reg.predict(X_train)) ** 0.5,
                                   100 * mean_squared_error(y_test, reg.predict(X_test)) ** 0.5)
    df_simple_train = pd.DataFrame({
        'Mean': [train_score],
        'Std. dev': [0],
        'CI: lower': [train_score],
        'CI: upper': [train_score],
    }, index=['Simple'])

    df_simple_test = pd.DataFrame({
        'Mean': [test_score],
        'Std. dev': [0],
        'CI: lower': [test_score],
        'CI: upper': [test_score],
    }, index=['Simple'])

    return df_simple_train, df_simple_test


def summary_box_plots(ticker, problem_type):
    df_list = []
    metric = 'accuracy' if problem_type == "classification" else 'rmse'
    metric_print = 'Accuracy' if problem_type == "classification" else 'RMSE'

    for algo in ENSEMBLES:
        df_tmp = pd.read_pickle(f"results/results_{algo} {problem_type} {ticker}.pkl")
        df_tmp.rename(columns={f'{metric}: train': f'train: {algo}',
                               f'{metric}: test': f'test: {algo}'}, inplace=True)
        df_list.append(df_tmp)

    for algo in STACKERS:
        df_tmp = pd.read_pickle(f"results/results_Stacker {algo} {problem_type} {ticker}.pkl")
        df_tmp.rename(columns={f'{metric}: train': f'train: {algo}',
                               f'{metric}: test': f'test: {algo}'}, inplace=True)
        df_list.append(df_tmp)

    df_concat = pd.concat(df_list, axis=1)
    simple_train, simple_test = simple_model_metrics(ticker, problem_type)

    df_concat_train = df_concat[[col for col in df_concat.columns if 'train' in col]]
    df_concat_test = df_concat[[col for col in df_concat.columns if 'test' in col]]

    df_concat_train = df_concat_train.rename(
        columns=dict(zip(['train: ' + name for name in ENSEMBLES], ENSEMBLES)))
    df_concat_train.rename(columns=dict(zip(['train: ' + stacker for stacker in STACKERS],
                                            [stacker + ' Stacker' for stacker in STACKERS])), inplace=True)

    df_concat_test = df_concat_test.rename(
        columns=dict(zip(['test: ' + name for name in ENSEMBLES], ENSEMBLES)))
    df_concat_test.rename(columns=dict(zip(['test: ' + stacker for stacker in STACKERS],
                                           [stacker + ' Stacker' for stacker in STACKERS])), inplace=True)

    fig1 = plt.figure(figsize=(8, 6))
    plt.title(f"Train {metric_print} for {ticker[1:]}", fontsize=22)
    df_concat_train.boxplot(fontsize=18, vert=False, )
    plt.vlines(simple_train['Mean'].values, 1, 8, colors="red", linestyles="dashed", label="simple")
    plt.xlabel(metric_print, fontsize=18)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.close()

    fig2 = plt.figure(figsize=(8, 6))
    plt.title(f"Test {metric_print} for {ticker[1:]}", fontsize=22)
    df_concat_test.boxplot(fontsize=18, vert=False)
    plt.xlabel(metric_print, fontsize=18)
    plt.vlines(simple_test['Mean'].values, 1, 8, colors="red", linestyles="dashed", label="simple")
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.close()

    return fig1, fig2


def hyperparam_table():
    param_grids = {'Random Forest': {'max_depth': [2, 4, 8], 'max_features': [2, 4, 8]},
                   'Extra Trees': {'max_depth': [2, 4, 8], 'max_features': [2, 4, 8]},
                   'Adaboost': {'learning_rate': [0.01, 0.1, 1.0, 5.0]},
                   'Gradient Boosting':
                       {'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [2, 4, 8], 'max_features': [2, 4, 8]},
                   'XGBoost': {'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [2, 4, 8]}}
    params_dict = dict(zip(param_grids.keys(), [list(param_grids[key].keys()) for key in param_grids.keys()]))
    output = pd.DataFrame({
        'Problem type': [],
        'Ticker': [],
        'Model': [],
        'Params': []
    })

    for problem_type in problem_types:
        for ticker in TICKERS:
            for ensemble in ENSEMBLES:
                data = load_data(ticker, problem_type)
                model = TreeBasedEnsemble(name=f"{ensemble} {problem_type} {ticker}",
                                          data=data, problem_type=problem_type, load_from_file=True)

                for param in params_dict[ensemble]:
                    all_params = model.model.get_params()
                    param_string = f'{param}: {all_params[param]}'
                    row = pd.DataFrame({
                        'Problem type': [problem_type],
                        'Ticker': [ticker[1:]],
                        'Model': [ENSEMBLES_SHORT_dict[ensemble]],
                        'Params': [param_string]
                    })
                    output = output.append(row)

            for stacker in STACKERS:
                if stacker == "Network":
                    network_params = np.loadtxt(f"models/Network {problem_type} {ticker}.txt")
                    param_strings_list = [f'neurons: {int(network_params[0])}',
                                          f'layers: {int(network_params[1])}',
                                          f'dropout: {network_params[2]}']
                    for param_string in param_strings_list:
                        row = pd.DataFrame({
                            'Problem type': [problem_type],
                            'Ticker': [ticker[1:]],
                            'Model': ["Network"],
                            'Params': [param_string]
                        })
                        output = output.append(row)

                if stacker == "SVM":
                    model = load(f"models/SVM {problem_type} {ticker}.joblib")
                    param_string = f"regularisation: {model.get_params()['C']}"

                    row = pd.DataFrame({
                        'Problem type': [problem_type],
                        'Ticker': [ticker[1:]],
                        'Model': ["SVM"],
                        'Params': [param_string]
                    })
                    output = output.append(row)

    output.reset_index(drop=True, inplace=True)
    output.rename(columns={'Params': 'Parameters'}, inplace=True)
    output['Model'] = np.where(output['Model'] == output['Model'].shift(), '', output['Model'])

    output.to_excel("results/hyperparameter.xlsx", index=False)
    output.to_pickle("results/hyperparameter.pkl")

    return output
