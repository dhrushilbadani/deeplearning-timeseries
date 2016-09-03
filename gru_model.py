from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math


class GRU_RNN:

    def __init__(self, look_back, batch_size = 1, dropout_probability = 0.2, init ='he_uniform', loss='mse', optimizer='rmsprop'):
        self.rnn = Sequential()
        self.batch_size = batch_size
        self.look_back = look_back
        self.rnn.add(GRU(4, input_dim=look_back, init=init))
        self.rnn.add(Dropout(dropout_probability))
        self.rnn.add(Dense(1, init=init))
        self.rnn.compile(loss=loss, optimizer=optimizer)

    def train(self, X, Y, nb_epoch=150):
        print('Training GRU-RNN...')
        self.rnn.fit(X, Y, nb_epoch=nb_epoch, batch_size=self.batch_size, verbose=2)

    def evaluate(self, X, Y):
        score = self.rnn.evaluate(X, Y, batch_size = self.batch_size, verbose=0)
        print score
        return score

    def predict(self, X):
        return self.rnn.predict(X)






