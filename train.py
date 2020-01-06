import os
import keras
import pandas as pd
import numpy as np
from models import LstmModel
from config import DIR_CONFIG, FILE_CONFIG


if __name__ == "__main__":
    # Initialize Dates
    # START_DATE = '2017-01-02'
    # END_DATE = '2019-06-01'
    START_DATE = '2015-07-06'
    END_DATE = '2018-05-28'

    # Initialize Model
    lstm = LstmModel(name='Old2')

    # Load List Of Companies
    companies = []

    with open(FILE_CONFIG["COMPANY_LIST"], 'r') as f:
        for line in f:
            companies.append(line.rstrip("\n"))
    print(companies)

    # Train Model With Different Companies Adj Close Values
    for company in companies:
        path = os.path.join(DIR_CONFIG["DATA_DIR"], '{}_data.csv'.format(company))
        raw_data = pd.read_csv(path, index_col=0)
        raw_data = raw_data.loc[:END_DATE]["Adj Close"].values
        raw_data = np.reshape(raw_data, (raw_data.shape[0], 1))
        print(raw_data.shape)
        lstm.train_model(company=company, raw_data=raw_data)
