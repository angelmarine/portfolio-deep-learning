import os
import json
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.externals import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(CURRENT_DIR, 'result')


def save_json(filename, records):
    filename = filename + '.txt'
    filepath = os.path.join(RESULT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(records, f)

    print("Saved Records To ", filepath)


def plot_performance(filename, backtest_dates, backtest_records):
    filename = filename + '.png'
    filepath = os.path.join(RESULT_DIR, filename)

    plt.figure(figsize=(10, 5))
    plt.title("Portfolio Performance")
    plt.plot(backtest_dates, backtest_records)
    plt.xlabel("Dates")
    plt.ylabel("Value($)")
    # plt.legend()
    plt.savefig(filepath)

    print("Saved Graph To ", filepath)


def save_checkpoint(model, folder='checkpoint', filename="lstm_model"):
    filename = filename + '.h5'
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder, filename)
    if not os.path.exists(folder):
        print("Checkpoint Directory Does Not Exist! Making Directory At {}".format(folder))
        os.mkdir(folder)
    else:
        print("Checkpoint Directory exists! ")
    model.save(filepath)
    print("Saved Model At ", filepath)


def load_checkpoint(folder='checkpoint', filename="lstm_model"):
    filename = filename + '.h5'
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder, filename)
    if not os.path.exists(folder):
        print("No Model In Path {}".format(folder))
    else:
        return load_model(filepath)


def save_scaler(scaler, folder='checkpoint', filename='lstm_model'):
    filename = filename + '.save'
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder, filename)
    if not os.path.exists(folder):
        print("Checkpoint Directory Does Not Exist! Making Directory At {}".format(folder))
        os.mkdir(folder)
    else:
        print("Checkpoint Directory Exists! ")
    joblib.dump(scaler, filepath)
    print("Saved Scaler At ", filepath)


def load_scaler(folder='checkpoint', filename='lstm_model'):
    filename = filename + '.save'
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder, filename)
    if not os.path.exists(folder):
        print("No Scaler In Path {}".format(folder))
    else:
        return joblib.load(filepath)