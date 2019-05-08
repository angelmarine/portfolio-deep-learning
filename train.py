import os
import keras
import pandas as pd
import numpy
from models import LstmModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'new_data')
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoint')

print("Data Stored In", DATA_DIR)
print("Checkpoint Stored In", CHECKPOINT_DIR)

if __name__ == "__main__":

    # Initialize Model
    lstm = LstmModel(name='Test1')

    # Load List Of Companies
    companies = []

    with open(os.path.join(CURRENT_DIR, "company_list.txt"), 'r') as f:
        for line in f:
            companies.append(line.rstrip("\n"))
    print(companies)

    # Train Model With Different Companies Data
    for company in companies:
        path = os.path.join(DATA_DIR, '{}_data.csv'.format(company))
        raw_data = pd.read_csv(path, index_col=0)
        raw_data = raw_data.loc[:'2018-05-25'].values
        #print(raw_data)
        lstm.train_model(raw_data)
