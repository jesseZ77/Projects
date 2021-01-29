from presentation_functions import *

ENSEMBLES = ['Random Forest', 'Extra Trees', 'Adaboost', 'Gradient Boosting', 'XGBoost']
ENSEMBLES_SHORT = ['RF', 'ET', 'ADA', 'GB', 'XGB']
TICKERS = ['^AXJO', '^GSPC', '^FTSE', '^N225']
TICKERS_PRINT = [ticker[1:] for ticker in TICKERS]


def main():
    """
    Runs functions that create tables and plots for the report
    """
    data_visualisation()
    write_hyper_params()
    write_result_tables()
    write_box_plots()


def data_visualisation():
    # Summary stats
    save_as_latex(summary_stats(), 'Summary Statistics: Daily Returns (\\%)', 'summary_stats')

    # Price and return visualisation
    for ticker in TICKERS:
        fig1, fig2 = price_return_plots(ticker)
        save_image(fig1, f'price_plot_{ticker}')
        save_image(fig2, f'return_plot_{ticker}')

    # Data description for the classification problem
    save_as_latex(cutoffs_table(), 'Classification: Class Cutoffs', 'cutoffs')
    save_as_latex(class_distributions(), 'Classification: Class Distributions', 'class_dist')

    # Data description for the regression problem
    df_train, df_test = regression_data_tables()
    save_as_latex(df_train, 'Regression: 5-Day Returns (\\%), train', 'regress_summary_stats_train')
    save_as_latex(df_test, 'Regression: 5-Day Returns (\\%), test', 'regress_summary_stats_test')


def write_hyper_params():
    df_hyperparams = hyperparam_table()
    for problem_type in problem_types:
        for ticker in TICKERS_PRINT:
            df_tmp = (df_hyperparams
                      [lambda x: x['Problem type'] == problem_type]
                      [lambda x: x['Ticker'] == ticker][['Model', 'Parameters']]
                      .set_index('Model'))

            caption = f'{problem_type.capitalize()} Hyper-parameters for {ticker}'
            file_name = f'hyperparams_{ticker}_{problem_type}'
            table = (df_tmp.to_latex(caption=caption, bold_rows=True))
            table = re.sub("\\\\begin{table}", "\\\\begin{table}[H]", table)
            text_file = open(f'Report/Tables/{file_name}.txt', "w")
            text_file.write(table)
            text_file.close()


def write_result_tables():
    for problem_type in ['classification', 'regression']:
        for ticker in TICKERS:
            ticker_print = ticker[1:]
            df_concat_train, df_concat_test = results_summary_table(ticker, problem_type)
            if problem_type == 'classification':
                caption = f'Accuracy: {ticker_print}'
            else:
                caption = f'RMSE: {ticker_print}'
            save_as_latex(df_concat_train, caption='Train ' + caption)
            save_as_latex(df_concat_test, caption='Test ' + caption)

        rank_summary_train, rank_summary_test = summarise_ranks(problem_type)
        if problem_type == 'classification':
            save_as_latex(rank_summary_train, caption='Classification Ranks: Train')
            save_as_latex(rank_summary_test, caption='Classification Ranks: Test')
        else:
            save_as_latex(rank_summary_train, caption='Regression Ranks: Train')
            save_as_latex(rank_summary_test, caption='Regression Ranks: Test')


def write_box_plots():
    for problem_type in ['classification', 'regression']:
        for ticker in TICKERS:
            fig1, fig2 = summary_box_plots(ticker, problem_type)
            save_image(fig1, f"boxplot train {problem_type} {ticker}")
            save_image(fig2, f"boxplot test {problem_type} {ticker}")


if __name__ == "__main__":
    main()
