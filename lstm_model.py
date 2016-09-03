from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
import numpy as np


class LSTM_RNN:

    def __init__(self, look_back, dropout_probability = 0.2, init ='he_uniform', loss='mse', optimizer='rmsprop'):
        self.rnn = Sequential()
        self.look_back = look_back
        self.rnn.add(LSTM(10, stateful = True, batch_input_shape=(1, 1, 1), init=init))
        self.rnn.add(Dropout(dropout_probability))
        self.rnn.add(Dense(1, init=init))
        self.rnn.compile(loss=loss, optimizer=optimizer)

    def batch_train_test(self, trainX, trainY, testX, testY, nb_epoch=150):
        print('Training LSTM-RNN...')
        for epoch in range(nb_epoch):
            print('Epoch '+ str(epoch+1) +'/{}'.format(nb_epoch))
            training_losses = []
            testing_losses = []
            for i in range(len(trainX)):
                y_actual = trainY[i]
                for j in range(self.look_back):
                    training_loss = self.rnn.train_on_batch(np.expand_dims(np.expand_dims(trainX[i][j], axis=1), axis=1),
                                                       np.array([y_actual]))
                    training_losses.append(training_loss)
                self.rnn.reset_states()

            print('Mean training loss = {}'.format(np.mean(training_losses)))

            mean_testing_loss = []
            for i in range(len(testX)):
                for j in range(self.look_back):
                    testing_loss = self.rnn.test_on_batch(np.expand_dims(np.expand_dims(testX[i][j], axis=1), axis=1),
                                                          np.array([testY[i]]))
                    testing_losses.append(testing_loss)
                self.rnn.reset_states()

                for j in range(self.look_back):
                    y_pred = self.rnn.predict_on_batch(np.expand_dims(np.expand_dims(testX[i][j], axis=1), axis=1))
                self.rnn.reset_states()

            mean_testing_loss = np.mean(testing_losses)
            print('Mean testing loss = {}'.format(mean_testing_loss))
        return mean_testing_loss







