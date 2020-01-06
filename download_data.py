import os
import pandas as pd
import yahoo_finance_pynterface as yahoo
from config import DIR_CONFIG, FILE_CONFIG


# Get Symbols From Archieve
symbols = []

with open(FILE_CONFIG["COMPANY_LIST"], 'r') as f:
    for line in f:
        symbols.append(line.rstrip("\n"))
print(symbols)

# START_DATE = '2017-01-02'
# END_DATE = '2019-12-30'

START_DATE = '2015-07-06'
END_DATE = '2018-12-31'

# Get Closing Prices From Yahoo
for symbol in symbols:
    data = yahoo.Get.Prices(symbol, period=[START_DATE, END_DATE])
    if data is None:
        print("No data for: ", symbol)
        continue

    # print(data)

    df = pd.DataFrame(data={
        'Open': data['Open'],
        'High': data['High'],
        'Low': data['Low'],
        'Adj Close': data['Adj Close'],
        'Close': data['Close'],
        'Volume': data['Volume']
    })

    date_series = df.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d"))
    df.insert(0, 'Date', date_series)
    # print(df)
    df = df.dropna()  # Drop NaN Rows
    # print(df)

    filepath = os.path.join(DIR_CONFIG["OLD_DATA_DIR"], '{}_data.csv'.format(symbol))
    # filepath = os.path.join(DIR_CONFIG["DATA_DIR"], '{}_data.csv'.format(symbol))
    df.to_csv(filepath, index=False)

    print("Downloaded And Saved Stock Data For: {} ".format(symbol))
