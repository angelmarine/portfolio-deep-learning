import os
import json
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.externals import joblib
from config import DIR_CONFIG


def save_json(filename, records):
    filename = filename + '.txt'
    filepath = os.path.join(DIR_CONFIG["RESULT_DIR"], filename)
    with open(filepath, 'w') as f:
        json.dump(records, f)

    print("Saved Records To ", filepath)


def plot_performance(filename, backtest_dates, backtest_records):
    filename = filename + '.png'
    filepath = os.path.join(DIR_CONFIG["RESULT_DIR"], filename)

    fig, ax = plt.subplots(1, 1)
    ax.title.set_text("Portfolio Performance")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Value($)")
    ax.plot(backtest_dates, backtest_records)
    ax.set_xticks(ax.get_xticks()[::25])
    plt.savefig(filepath)

    print("Saved Graph To ", filepath)


def save_checkpoint(model, filename="lstm_model"):
    filename = filename + '.h5'
    filepath = os.path.join(DIR_CONFIG["CHECKPOINT_DIR"], filename)

    if not os.path.exists(DIR_CONFIG["CHECKPOINT_DIR"]):
        print("Checkpoint Directory Does Not Exist! Making Directory At {}".format(DIR_CONFIG["CHECKPOINT_DIR"]))
        os.mkdir(DIR_CONFIG["CHECKPOINT_DIR"])
    else:
        print("Checkpoint Directory exists! ")
    model.save(filepath)
    print("Saved Model At ", filepath)


def load_checkpoint(filename="lstm_model"):
    filename = filename + '.h5'
    filepath = os.path.join(DIR_CONFIG["CHECKPOINT_DIR"], filename)

    if not os.path.exists(DIR_CONFIG["CHECKPOINT_DIR"]):
        print("No Model In Path {}".format(DIR_CONFIG["CHECKPOINT_DIR"]))
    else:
        return load_model(filepath)


def save_scaler(scaler, filename='lstm_model'):
    filename = filename + '.save'
    filepath = os.path.join(DIR_CONFIG["CHECKPOINT_DIR"], filename)

    if not os.path.exists(DIR_CONFIG["CHECKPOINT_DIR"]):
        print("Checkpoint Directory Does Not Exist! Making Directory At {}".format(DIR_CONFIG["CHECKPOINT_DIR"]))
        os.mkdir(DIR_CONFIG["CHECKPOINT_DIR"])
    else:
        print("Checkpoint Directory Exists! ")
    joblib.dump(scaler, filepath)
    print("Saved Scaler At ", filepath)


def load_scaler(filename='lstm_model'):
    filename = filename + '.save'
    filepath = os.path.join(DIR_CONFIG["CHECKPOINT_DIR"], filename)

    if not os.path.exists(DIR_CONFIG["CHECKPOINT_DIR"]):
        print("No Scaler In Path {}".format(DIR_CONFIG["CHECKPOINT_DIR"]))
    else:
        return joblib.load(filepath)