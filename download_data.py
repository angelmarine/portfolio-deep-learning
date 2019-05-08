import os
import pandas as pd
import yahoo_finance_pynterface as yahoo

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'old_data')

# Get Symbols From Archieve
symbols = []

with open(os.path.join(CURRENT_DIR, "company_list.txt"), 'r') as f:
    for line in f:
        symbols.append(line.rstrip("\n"))
print(symbols)

START_DATE = '2015-01-02'
END_DATE = '2018-06-30'

# START_DATE = '2015-07-06'
# END_DATE = '2018-12-31'

# Get Closing Prices From Yahoo
for symbol in symbols:
    data = yahoo.Get.Prices(symbol, period=[START_DATE, END_DATE])
    if data is None:
        print("No data for: ", symbol)
        continue

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
    filepath = os.path.join(DATA_DIR, '{}_data.csv'.format(symbol))
    df.to_csv(filepath, index=False)
    print("Downloaded And Saved Stock Data For: {} ".format(symbol))
