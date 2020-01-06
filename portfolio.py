import os
import keras
import pandas as pd
import numpy as np
import datetime
from utils import plot_performance, save_json
from models import LstmModel
from config import DIR_CONFIG, FILE_CONFIG


def evaluate_portfolio(date, portfolio, companies_data):
    # Get Current Prices For Companies In Portfolio
    prices = {}

    for company in portfolio:
        data = companies_data[company]
        try:
            idx = data[data['Date'] == date].index.values[0]
        except IndexError:  # Use The Last Valid Previous Data When There Is Empty Data
            dates = pd.date_range(end=date, freq='D', periods=10).strftime("%Y-%m-%d")
            print(dates)
            idx = data[data['Date'].isin(dates)].index.values[-1]
        prices[company] = data.iloc[idx, 4]  # Adj Closing Price

    # Evaluate Portfolio
    evaluation = 0

    for company, numbers in portfolio.items():
        evaluation = evaluation + (prices[company] * numbers)

    return evaluation


''' Calculate Return-To-Volatility Ratio Using Close Price
'''


def calculate_rtv(data, windows_size=5):
    # print(data)
    window = "{}D".format(windows_size)
    data_pct = data.pct_change()
    data_weekly_pct = data_pct.resample(window, label='right').mean().fillna(0)
    data_weekly_sd = data_pct.resample(window, label='right').std(ddof=0).fillna(0)
    rtv = data_weekly_pct.iloc[0, 0] / data_weekly_sd.iloc[0, 0]
    return rtv


def uniform_portfolio(date, asset, companies_data):
    # Get All Valid Companies Prices For That Date
    companies_prices = {}
    for company, data in companies_data.items():
        # Current Date's Index
        try:
            idx = data[data['Date'] == date].index.values[0]
        except IndexError:  # When Company Data Is Empty On That Date
            continue
        companies_prices[company] = data.iloc[idx, 4]

    num_companies = len(companies_prices.keys())
    portfolio_pct = {company: (1 / num_companies) for company in companies_prices}
    portfolio = {company: ((asset * pct) / companies_prices[company]) for company, pct in portfolio_pct.items()}
    return portfolio, portfolio_pct


def random_portfolio(date, asset, companies_data):
    # Get All Valid Companies Prices For That Date
    companies_prices = {}
    for company, data in companies_data.items():
        # Current Date's Index
        try:
            idx = data[data['Date'] == date].index.values[0]
        except IndexError:  # When Company Data Is Empty On That Date
            continue
        companies_prices[company] = data.iloc[idx, 4]

    num_companies = len(companies_prices.keys())
    random_list = np.random.randint(low=1, high=100, size=num_companies)
    sum_random_list = sum(random_list)
    portfolio_pct = {company: random_list[idx] / sum_random_list for idx, company in enumerate(companies_prices.keys())}
    portfolio = {company: ((asset * pct) / companies_prices[company]) for company, pct in portfolio_pct.items()}
    return portfolio, portfolio_pct


''' Construct Portfolio Based On Predictions Of Prices One-Week Ahead
'''


def rebalance_porfolio(date, asset, companies_data):
    # date_string = datetime.datetime.strftime(date, "%Y-%m-%d")
    print("=" * 50)
    print("Rebalancing Portfolio For Current Date: ", date)
    print("=" * 50)

    # Load Trained Model
    lstm = LstmModel(name="Old1", load=True)

    # Save Values In Dict
    companies_rtv = dict()
    companies_prices = dict()

    # Predict One-Week Price
    # Calculate Return-To-Volatility
    # Save The Result For Different Companies
    day_in, day_out, num_features = lstm.load_settings()

    for company, data in companies_data.items():
        # Current Date's Index
        try:
            idx = data[data['Date'] == date].index.values[0]
        except IndexError:  # When Company Data Is Empty On That Date
            continue

        # Predictions From LSTM
        _input = data.iloc[idx - day_in:idx]['Adj Close'].values
        _input = np.reshape(_input, (_input.shape[0], 1))
        predictions = pd.DataFrame(lstm.predict(company=company, _input=_input),
                                   index=pd.date_range(start=date, periods=day_out))

        # Calculate Return-To-Volatility
        rtv = calculate_rtv(predictions)
        companies_rtv[company] = rtv

        # Get Adj Closing Price List
        companies_prices[company] = data.iloc[idx, 4]

    # Sort The Results Descending & Choose Top 10 Companies
    filtered_list = list(filter(lambda x: x[1] > 0, companies_rtv.items()))
    sorted_list = sorted(filtered_list, key=lambda x: x[1], reverse=True)
    companies_rtv_filter = {t[0]: t[1] for t in sorted_list[:10]}

    # Build Portfolio
    rtv_sum = sum(companies_rtv_filter.values())
    portfolio_pct = {key: (value / rtv_sum) for key, value in companies_rtv_filter.items()}
    portfolio = {company: ((asset * pct) / companies_prices[company]) for company, pct in portfolio_pct.items()}

    print("Company Prices: ", companies_prices)
    print("Companies Return-To-Volatility: ", companies_rtv)
    print("Filtered 10 Companies: ", companies_rtv_filter)
    print("Portfolio Percentage: ", portfolio_pct)
    print("Portfolio: ", portfolio)

    return portfolio, portfolio_pct


def backtest(start_date, end_date, start_asset, mode="lstm"):
    if mode == "uniform":
        rebalance = uniform_portfolio
    elif mode == "random":
        rebalance = random_portfolio
    else:
        rebalance = rebalance_porfolio

    # ================= Initialize Configurations ====================
    # 1. Set Start Asset
    # 2. Set Back Test Dates
    # 3. Initialize Back Test Records
    # 4. Load Companies List
    # 5. Load Companies Data

    # 1. Set Start Asset
    current_asset = start_asset
    current_portfolio = dict()

    # 2. Set Back Test Dates
    day_out = 5
    path = os.path.join(DIR_CONFIG["DATA_DIR"], 'AAPL_data.csv')
    df = pd.read_csv(path)
    df = df[(df["Date"] > start_date) & (df["Date"] <= end_date)]
    backtest_dates = df["Date"].values
    print("Back Test Dates: ", backtest_dates)

    # 3. Initialize Back Test Records
    benchmark_records = []
    backtest_records = []
    portfolio_records = []

    # 4. Load Companies List
    companies_list = []
    with open(FILE_CONFIG["COMPANY_LIST"], 'r') as f:
        for line in f:
            companies_list.append(line.rstrip("\n"))

    # 5. Load All Companies Data
    companies_data = dict()
    for company in companies_list:
        path = os.path.join(DIR_CONFIG["DATA_DIR"], '{}_data.csv'.format(company))
        companies_data[company] = pd.read_csv(path)

    # ================================================================

    print("======================Start Backtest==========================")
    print("Start Date: ", start_date)
    print("End Date: ", end_date)
    print("Start Asset: ", start_asset)
    print("Mode: ", mode)

    # Start Back Test
    for i in range(len(backtest_dates)):
        # Rebalance Portfolio Every 5 Days
        if (i % 5) == 0 and (i + day_out) < len(backtest_dates):
            current_portfolio, percentages_portfolio = rebalance(date=backtest_dates[i], asset=current_asset,
                                                                 companies_data=companies_data)
            portfolio_records.append(percentages_portfolio)
        # Evaluate Porfolio Every Day
        current_asset = evaluate_portfolio(date=backtest_dates[i], portfolio=current_portfolio,
                                           companies_data=companies_data)
        backtest_records.append(current_asset)

        print("Current Asset Value: ", current_asset)
        print("Records: ", backtest_records)

    # Plot Portfolio Performance Graph
    plot_performance(filename='portfolio_graph_{}_{}'.format(mode, start_date), backtest_dates=backtest_dates,
                     backtest_records=backtest_records)

    # Save Performance and Portfolio Records
    save_json(filename='portfolio_performance_{}_{}'.format(mode, start_date), records=backtest_records)
    save_json(filename='portfolio_records_{}_{}'.format(mode, start_date), records=portfolio_records)


if __name__ == "__main__":
    # START_DATE = "2018-01-01"
    # END_DATE = "2018-05-28"
    START_DATE = "2018-05-29"
    END_DATE = "2018-12-28"
    backtest(start_date=START_DATE, end_date=END_DATE, start_asset=10000, mode='lstm')
    backtest(start_date=START_DATE, end_date=END_DATE, start_asset=10000, mode='uniform')
    backtest(start_date=START_DATE, end_date=END_DATE, start_asset=10000, mode='random')
