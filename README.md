# Timeseries Forecasting with Deep Learning
This Python project uses LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) based Recurrent Neural Networks to forecast (predict) timeseries using Keras + Theano. We compare the results produced by each of these deep neural networks with those from a linear regression model.

<b>Dataset:</b> <a href = "https://datamarket.com/data/set/235j/number-of-daily-births-in-quebec-jan-01-1977-to-dec-31-1990#!ds=235j&display=line"> Number of daily births in Quebec, Jan. 01 '77 - Dec. 31 '90 (Hipel & McLeod, 1994) </a>

##Usage
I suggest you install [Virtualenv](https://virtualenv.pypa.io/en/stable/) before trying this out. You will need [matplotlib](matplotlib.org) for the plotting as well (this doesn't work too well with [virtual environments](http://matplotlib.org/faq/virtualenv_faq.html), though).
```
git clone https://github.com/dhrushilbadani/deeplearning-timeseries.git
cd deeplearning-timeseries
virtualenv ENV
source ENV/bin/activate
pip install --upgrade pip
pip install keras h5py pandas sklearn
python evaluate.py
```


##Architecture & Model Properties
We use Keras' [Sequential](https://keras.io/models/sequential/) model to construct a recurrent neural network. There are only two layers:
<ul>
<li> Layer 1 : Either a LSTM (with output dimension 4,  and statefulness enabled) layer or a GRU (with output dimension 4) layer.
<li> Layer 2 : A fully-connected Dense Layer with output dimension 1. </li>
<li> Default optimizer: Adam; Default # of epochs: 100. </li>
<li> Feature Normalization: 0-1 Minmax Normalization. </li>
<li> Accuracy Metric: Root Mean Squared Error. </li>
</ul>
<i>This architecture can certainly further be optimized - I just haven't had the chance to experiment thanks to my laptop's constraints! </i>

##Results & Observations
1. The GRU-RNN model performed the best with a RMSE of 156.413731922 (look back = 34).
2. Naively making the RNN "deeper" did not yield immediate results; I didn't fine-tune the parameters (output_dim, for example) though. 
3. Making the LSTM network stateful (setting ```stateful=true``` when initializing the LSTM layer) did yield a slight performance improvement though. 
4. Using Glorot initializations yielded a performance improvement. However, using He uniform initialization (Gaussian initialization scaled by fan_in) yielded even better results than with Glorot.

##Files

<li>```data/number-of-daily-births-in-quebec.csv``` : Dataset. </li>
<li> ```rnnmodel.py```: Model for LSTM/GRU-based Recurrent Neural Networks. </li>
<li>  ```evaluate.py```: Loads and preprocesses the dataset, creates LSTM-RNN, GRU-RNN and Linear Regression models, and outputs results. </li>



