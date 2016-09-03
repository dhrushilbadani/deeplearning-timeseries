import numpy
from gru_model import GRU_RNN
from lstm_model import LSTM_RNN
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
import warnings
import matplotlib.pyplot as plt

# load dataset at path into a Pandas dataframe
def load_dataset(path):
    dataframe = pandas.read_csv(path, usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values.astype(int)
    return dataset

# convert an array of values into a dataset matrix
def matricise_dataset(dataset, look_back=1, batch_size=1):
    n = int(math.floor((len(dataset)-look_back-1)/batch_size)) * batch_size # the number of samples must be divisible by the batch size
    dataX, dataY = [], []
    for i in range(n):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def matricise_dataset_lstm(dataset, look_back=1, batch_size=1):
    n = len(dataset)
    dataX, dataY = [], []
    i = 0
    while (i + look_back) < n:
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
        i = i + look_back + 1
    return numpy.array(dataX), numpy.array(dataY)


# split into train and test sets
def split_data(look_back, batch_size):
    lstm_trainX, lstm_trainY = matricise_dataset_lstm(train, look_back=look_back, batch_size=batch_size)
    lstm_testX, lstm_testY = matricise_dataset_lstm(test, look_back=look_back, batch_size=batch_size)
    lstm_trainX = numpy.reshape(lstm_trainX, (len(lstm_trainX), look_back, 1))
    lstm_trainY = numpy.reshape(lstm_trainY, (len(lstm_trainY), 1))
    lstm_testX = numpy.reshape(lstm_testX, (len(lstm_testX), look_back, 1))
    trainX, trainY = matricise_dataset(train, look_back=look_back)
    testX, testY = matricise_dataset(test, look_back=look_back)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return lstm_trainX, lstm_trainY, lstm_testX, lstm_testY, trainX, trainY, testX, testY

# evaluate GRU, LSTM and linear regression models
def evaluate_models(dataset, look_back, batch_size):
    print 'Processing dataset for lookback = ' + str(look_back) + '...'
    lstm_trainX, lstm_trainY, lstm_testX, lstm_testY, trainX, trainY, testX, testY = split_data(look_back, batch_size)
    print 'Training & evaluating GRU-RNN, LSTM-RNN and regression models for lookback = ' + str(look_back) + '...'
    gru_model = GRU_RNN(look_back, batch_size)
    gru_model.train(trainX, trainY)
    gru_test_mse = gru_model.evaluate(testX, testY)
    lstm_model = LSTM_RNN(look_back)
    lstm_test_mse = lstm_model.batch_train_test(lstm_trainX, lstm_trainY, lstm_testX, lstm_testY)
    nsamples, nx, ny = trainX.shape
    regr_trainX = trainX.reshape((nsamples,nx*ny))
    nsamples, nx, ny = testX.shape
    regr_testX = testX.reshape((nsamples,nx*ny))
    regr = linear_model.LinearRegression()
    regr.fit(regr_trainX, trainY)
    regr_mse = numpy.mean((regr.predict(regr_testX) - testY) ** 2)
    print("With lookback = " + str(look_back) + ", Regression Score: %.2f"
          % regr_mse)
    print 'Completed model evaluation for lookback = ' + str(look_back) + '...'
    return gru_test_mse, lstm_test_mse, regr_mse

# parse mse values and print the best results for each models (GRU, LSTM, Regression)
def print_results(mse_vals):
    print 'Completed model evaluation for all lookback values...'
    gru_mse_vals = [x[0] for x in mse_vals]
    lstm_mse_vals = [x[1] for x in mse_vals]
    regr_mse_vals = [x[2] for x in mse_vals]
    gru_mse_min = min(gru_mse_vals)
    gru_mse_argmin = numpy.argmin(gru_mse_vals) + 1
    lstm_mse_min = min(lstm_mse_vals)
    lstm_mse_argmin = numpy.argmin(lstm_mse_vals) + 1
    regr_mse_min = min(regr_mse_vals)
    regr_mse_argmin = numpy.argmin(regr_mse_vals) + 1
    print 'Best mse with a GRU recurrent neural network was ' + str(gru_mse_min) + ' with a look back of ' + str(gru_mse_argmin)
    print 'Best mse with a LSTM recurrent neural network was ' + str(lstm_mse_min) + ' with a look back of ' + str(lstm_mse_argmin)
    print 'Best mse with a linear regression model was ' + str(regr_mse_min) + ' with a look back of ' + str(regr_mse_argmin)

warnings.filterwarnings("ignore")
numpy.random.seed(7)
DATA_PATHS = ['data/number-of-daily-births-in-quebec.csv']
TRAIN_RATIO = 0.80
dataset = load_dataset(DATA_PATHS[0])
train_size = int(len(dataset) * TRAIN_RATIO)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
mse_vals = []
batch_size = 1
for l in range(1, 40):
    mse_vals.append(evaluate_models(dataset, l, batch_size))
print_results(mse_vals)



