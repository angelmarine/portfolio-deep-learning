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
        self.day_in = 50
        self.step = 2  # these means use previous 50 days to predict the future 2 days
        self.day_want = 10  # this is the number of days you want to predict
        self.num_features = 6

        if load:
            self.name = name
            self.model = load_checkpoint(filename=self.name)
            self.scaler = load_scaler(filename=self.name)
        else:
            self.name = name
            self.model = self.build_model()
            self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def build_model(self):
        keras.backend.clear_session()
        model = Sequential()

        model.add(LSTM(60, input_shape=(self.day_in, self.num_features)))
        model.add(RepeatVector(2))
        model.add(LSTM(20, return_sequences=True))
        model.add(TimeDistributed(Dense(self.num_features)))

        model.compile(loss='mse', optimizer='rmsprop')
        
        print("Built Model With Following Structure")
        print(model.summary())
        return model
    
    def train_model(self, raw_data):
        # Process (Split + Normalize) Raw Data To Train and Label Data
        normalized_data = self.scaler.fit_transform(raw_data)
        train_data = []
        label_data = []
        for i in range(self.day_in, normalized_data.shape[0] - 1 - self.day_want):
            train_data.append(normalized_data[i - self.day_in:i, :])
            label_data.append(normalized_data[i:i + self.step, :])

        # Train Data
        self.model.fit(x=np.array(train_data), y=np.array(label_data), epochs=200, batch_size=512)

        # Save Model And Scaler
        save_checkpoint(self.model, filename=self.name)
        save_scaler(self.scaler, filename=self.name)

    ''' Input should have length of (day_in + day_want - step + 1) = 59
    '''
    def predict(self, _input):
        normalized_data = self.scaler.fit_transform(_input)
        whole_pred = np.zeros((self.day_want, self.num_features))
        for i in range(0, self.day_want, self.step):
            test_data = normalized_data[i:normalized_data.shape[0] - self.day_want + i + 1, :]
            test_data = np.reshape(test_data, (1, test_data.shape[0], self.num_features))
            step_pred = self.model.predict(test_data)
            #print("step_pred", step_pred.shape)
            #print(step_pred)
            step_pred = np.reshape(step_pred, (self.step, self.num_features))
            step_pred = self.scaler.inverse_transform(step_pred)
            whole_pred[i:i + self.step, :] = step_pred
        return whole_pred


