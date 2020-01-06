"""
LSTM model for prediction
input: the download stock data from internet, should include the days you want to predict
output: the prediction result of the days you want to predict
"""

import os
import keras
import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import load_model
from utils import save_checkpoint, load_checkpoint, save_scaler, load_scaler


class LstmModel:
    def __init__(self, name, load=False):
        # Model Training Configuration
        self.day_in = 50  # number of input days
        self.day_out = 5  # number of output days
        # self.day_want = 10  # this is the number of days you want to predict
        self.num_features = 1

        if load:
            self.name = name
            self.model = load_checkpoint(filename=self.name)
            self.scaler = load_scaler(filename=self.name)
        else:
            self.name = name
            self.model = self.build_model()
            # self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler = {}

    def load_settings(self):
        return self.day_in, self.day_out, self.num_features

    def build_model(self):
        keras.backend.clear_session()
        model = Sequential()

        model.add(LSTM(60, input_shape=(self.day_in, self.num_features)))
        model.add(RepeatVector(self.day_out))
        model.add(LSTM(20, return_sequences=True))
        model.add(TimeDistributed(Dense(self.num_features)))

        model.compile(loss='mse', optimizer='adam')

        print("Built Model With Following Structure")
        print(model.summary())
        return model

    def train_model(self, company, raw_data, epochs=100):
        # Process (Split + Normalize) Raw Data To Train and Label Data
        try:
            scaler = self.scaler[company]
        except KeyError:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler[company] = scaler

        normalized_data = scaler.fit_transform(raw_data)
        # print(normalized_data)
        train_data = []
        label_data = []
        for i in range(normalized_data.shape[0] - self.day_in - self.day_out):
            train_data.append(normalized_data[i:i + self.day_in])
            label_data.append(normalized_data[i + self.day_in:i + self.day_in + self.day_out])
            # train_data.append(normalized_data[i - self.day_in:i, :])
            # label_data.append(normalized_data[i:i + self.day_out, :])

        # print("Train Data", train_data)
        # print("Label Data", label_data)

        # Train Data
        self.model.fit(x=np.array(train_data), y=np.array(label_data), epochs=epochs, batch_size=128)

        # Save Model And Scaler
        save_checkpoint(self.model, filename=self.name)
        save_scaler(self.scaler, filename=self.name)

    def predict(self, company, _input):
        try:
            scaler = self.scaler[company]
        except KeyError:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler[company] = scaler

        normalized_data = scaler.fit_transform(_input)
        normalized_data = np.reshape(normalized_data, (1, normalized_data.shape[0], self.num_features))
        pred = self.model.predict(normalized_data)
        # print("Predictions For ", company, pred)
        pred = np.reshape(pred, (self.day_out, self.num_features))
        pred = scaler.inverse_transform(pred)
        return pred

    '''
    def predict(self, _input):
        whole_pred = np.zeros((self.day_want, self.num_features))
        for i in range(0, self.day_want, self.day_out):
            test_data = normalized_data[i:normalized_data.shape[0] - self.day_want + i + 1, :]
            test_data = np.reshape(test_data, (1, test_data.shape[0], self.num_features))
            step_pred = self.model.predict(test_data)
            # print("step_pred", step_pred.shape)
            # print(step_pred)
            step_pred = np.reshape(step_pred, (self.day_out, self.num_features))
            step_pred = self.scaler.inverse_transform(step_pred)
            whole_pred[i:i + self.day_out, :] = step_pred
        return whole_pred
    '''


