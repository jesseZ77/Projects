from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import os


def download_data(ticker, start, end):
    """
    Downloads data from Yahoo Finance, processes and then saves to file
    """
    if not os.path.exists(f'data/{ticker}'):
        os.makedirs(f'data/{ticker}')
    yahoo_financials = YahooFinancials(ticker)
    historical_stock_prices = yahoo_financials.get_historical_price_data(start, end, 'daily')
    data = pd.DataFrame(historical_stock_prices[ticker]['prices'])[['formatted_date', 'adjclose']]
    data = (data
            .rename(columns={'formatted_date': 'date', 'adjclose': 'price'})
            .assign(date=lambda x: pd.to_datetime(x['date'], format="%Y-%m-%d"),
                    returns=lambda x: np.log(x['price'] / x['price'].shift()))
            .dropna())
    data.to_csv(f'data/{ticker}/data.csv')
    data.to_pickle(f'data/{ticker}/data.pkl')


def make_datasets(ticker, n_trailing=20, n_step_ahead=5, test_size=0.3,
                  make_classification=True, make_regression=True, make_multistep=True):
    """
    Creates datasets for analysis:
        - Splits data into train and test sets
        - Creates features and targets
        - Writes created datasets to file
    """
    data = pd.read_csv(f'data/{ticker}/data.csv')
    returns = data.returns.values
    test_cutoff = int(len(returns) * (1 - test_size))

    return_splits = dict(
        zip(('train', 'test'),
            np.split(returns, [test_cutoff])))

    for k, v in return_splits.items():
        X = np.array([v[x: x + n_trailing]
                      for x in range(len(v) - n_trailing - n_step_ahead + 1)])
        np.savetxt(f"data/{ticker}/X_{k}.txt", X)

        y_raw = np.array([v[x + n_trailing: x + n_trailing + n_step_ahead]
                          for x in range(len(v) - n_trailing - n_step_ahead + 1)])
        y_regression = np.sum(y_raw, axis=1)

        if k == "train":
            down_cut, up_cut = np.quantile(y_regression, (1/3, 2/3))
            print(down_cut, up_cut)

        if make_regression:
            np.savetxt(f"data/{ticker}/y_{k}_regression.txt", y_regression)

        if make_classification:
            y_classification = np.select([y_regression < down_cut, y_regression < up_cut], [0, 1], 2)
            np.savetxt(f"data/{ticker}/y_{k}_classification.txt", y_classification)

        if make_multistep:
            np.savetxt(f"data/{ticker}/y_{k}_multistep.txt", y_raw)

    return down_cut, up_cut


def main():
    start = '2001-01-01'
    end = '2020-11-01'

    cutoffs = []

    for ticker in ['^AXJO', '^GSPC', '^FTSE', '^N225']:
        download_data(ticker, start, end)
        cutoffs.append(make_datasets(ticker, n_trailing=20, make_multistep=False))
    np.savetxt('data/cutoff.txt', np.array(cutoffs))


if __name__ == "__main__":
    main()
