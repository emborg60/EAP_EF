from math import sqrt
from math import exp
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
data = pd.read_csv('Data/All_Merged.csv')
data.isna().sum()
# Inserting 0 for NA
data.fillna(0, inplace=True)
data = data.set_index('date')
values = data.values

# View data
groups = [0, 1, 2, 3, 4, 5, 6]
i = 1
# plot each column
pyplot.figure(figsize = (10,10))
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(data.columns[group], y=0.5, loc='left')
    i += 1
pyplot.show()

# Differencing the data. All series become stationary once they have been differenced.
diff_data = data.diff()
values = diff_data.values

# View differenced, stationary data
# View data
i = 1
# plot each column
pyplot.figure(figsize = (10,10))
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(diff_data.columns[group], y=0.5, loc='left')
    i += 1
pyplot.show()

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
lags = 1
reframed = series_to_supervised(scaled, lags, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[7*(lags+1)-6, 7*(lags+1)-5, 7*(lags+1)-4, 7*(lags+1)-3, 7*(lags+1)-2, 7*(lags+1)-1]]
              , axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = 2901
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# Som i boosting - men uden validation, kun med rolling window
pred_day = 2901  # Predict for this day, for the next H-1 days. Note indexing of days start from 0.

H = 30  # Forecast horizon, in days. Note there are about 252 trading days in a year
train_size = int(365 * 0.75)
val_size = int(365 * 0.25)
train_val_size = train_size + val_size  # Size of train+validation set
print("No. of days in train+validation set = " + str(train_val_size))

####### OLD
# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]
# # reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#######

# design network
# Rules of thumb for hidden nodes and dropout:
# https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046

pred = pd.DataFrame()

while data.index[pred_day] < data.index[len(data) - 1]:
    # NEW: split into input and outputs
    train_X, train_y = values[pred_day - train_val_size:pred_day, :-1], values[pred_day - train_val_size:pred_day, -1]
    test_X, test_y = values[pred_day:pred_day + H, :-1], values[pred_day:pred_day + H, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    hidden_node = int(2/3 * train_X.shape[1] * train_X.shape[2])
    model = Sequential()
    model.add(LSTM(hidden_node, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=7, validation_data=(test_X, test_y), verbose=2,
                         shuffle=False)
    append = pd.DataFrame(model.predict(test_X))
    pred = pd.concat([pred, append], ignore_index=True)

    pred_day = pred_day + H


# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# make a prediction
yhat = pred
test_X = scaled[-len(yhat):,1:]  # test[:, :-1]
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X), axis=1)  # [:, 1:]
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert differencing for forecast
actual = Series(data['price'].tail(len(inv_yhat) + 1)).iloc[:-1]
sc_inv_yhat = inv_yhat + actual

# invert scaling for actual
test_y = pd.DataFrame(scaled[-len(yhat):,0])  # pd.DataFrame(test[:, -1])
# test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X), axis=1) # [:, 1:]
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# Invert differencing for actual
sc_inv_y = inv_y + actual

# calculate RMSE
rmse = sqrt(mean_squared_error(sc_inv_y, sc_inv_yhat))
print('Test RMSE: %.3f' % rmse)

final_data = pd.DataFrame()
pred = pd.DataFrame()
pred['LSTM'] = sc_inv_yhat
pred.index = data.iloc[-len(yhat):,0].index
final_data['Price'] = data['price']
final_data = final_data.merge(pred, how='left', left_index=True, right_index=True)

_ = final_data[['Price', 'LSTM']].plot(figsize=(10, 4))

# Only predictions
Testing_Period = pd.DataFrame()
Testing_Period['Price'] = sc_inv_y
Testing_Period['LSTM'] = sc_inv_yhat
Testing_Period.index = data.iloc[-len(yhat):,0].index
_ = Testing_Period[['Price', 'LSTM']].plot(figsize=(10, 4))

# Including a Naive forecast
naive = pd.DataFrame()
naive['t'] = data['price']
naive['naive'] = naive['t'].shift(1)
naive = naive.iloc[n_train_hours+1:-1, :]
naive_pred = naive['naive']
naive = naive['naive']

# calculate RMSE for Naive forecast
rmse = sqrt(mean_squared_error(sc_inv_y, naive_pred[2:]))
print('Test RMSE: %.3f' % rmse)

final_data = final_data.merge(naive, how='left', left_index=True, right_index=True)

_ = final_data[['Price', 'LSTM', 'naive']].plot(figsize=(10, 4))

# Export predictions
sc_inv_yhat.to_csv(r'Data\LSTM_Pred.csv')



# Grid search experiment
from itertools import product
pred_day = 2901
neuron = [2, 4, 6]
batch = [7, 14]
epoch = [100, 200]
dropout = [0, 0.2]
parameters = product(neuron, dropout, epoch, batch)
parameters_list = list(parameters)
len(parameters_list)

losses = pd.DataFrame()
j = 1
pred = pd.DataFrame()

while data.index[pred_day] < data.index[len(data) - 1]:
    # NEW: split into input and outputs
    train_X, train_y = values[pred_day - train_val_size:pred_day - val_size, :-1], \
                       values[pred_day - train_val_size:pred_day - val_size, -1]
    val_X, val_y = values[pred_day - val_size:pred_day, :-1], \
                   values[pred_day - val_size:pred_day, -1]
    train_val_X, train_val_y = values[pred_day - train_val_size:pred_day, :-1], \
                               values[pred_day - train_val_size:pred_day, -1]
    test_X, test_y = values[pred_day:pred_day + H, :-1], \
                     values[pred_day:pred_day + H, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
    train_val_X = train_val_X.reshape((train_val_X.shape[0], 1, train_val_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    i = 1
    best_loss = float("inf")
    for param in parameters_list:
        # hidden_node = int(2/3 * train_X.shape[1] * train_X.shape[2])
        model = Sequential()
        model.add(LSTM(param[0], input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(param[1]))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        # # fit network
        history = model.fit(train_X, train_y, epochs=param[2], batch_size=param[3], validation_data=(val_X, val_y), verbose=0,
                         shuffle=False)
        # losses = losses.append(history.history['val_loss'])
        name = str(j) + ',' + str(param)
        # losses[name] = history.history['val_loss']
        loss = history.history['val_loss'][-1]

        print ("Percent finished with grid search for current time prediction:")
        print(round(i/len(parameters_list),2))
        i = i + 1
        if loss < best_loss:
            best_loss = loss
            best_param = param
            print("New best parameters:")
            print(best_param)
            print("_______")

    # Fitting the chosen model
    model = Sequential()
    model.add(LSTM(best_param[0], input_shape=(train_val_X.shape[1], train_val_X.shape[2])))
    model.add(Dropout(best_param[1]))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # # fit network
    history = model.fit(train_val_X, train_val_y, epochs=best_param[2], batch_size=best_param[3], validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # predict using best model grid searched model to test data
    append = pd.DataFrame(model.predict(test_X))
    pred = pd.concat([pred, append], ignore_index=True)

    j = j+1
    print(j)
    pred_day = pred_day + H


# make a prediction
yhat = pred
test_X = scaled[-len(yhat):,1:]  # test[:, :-1]
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X), axis=1)  # [:, 1:]
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert differencing for forecast
actual = Series(data['price'].tail(len(inv_yhat) + 1)).iloc[:-1]
sc_inv_yhat = inv_yhat + actual

# invert scaling for actual
test_y = pd.DataFrame(scaled[-len(yhat):,0])  # pd.DataFrame(test[:, -1])
# test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X), axis=1) # [:, 1:]
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# Invert differencing for actual
sc_inv_y = inv_y + actual

# calculate RMSE
rmse = sqrt(mean_squared_error(sc_inv_y, sc_inv_yhat))
print('Test RMSE: %.3f' % rmse)

final_data = pd.DataFrame()
pred = pd.DataFrame()
pred['LSTM'] = sc_inv_yhat
pred.index = data.iloc[-len(yhat):,0].index
final_data['Price'] = data['price']
final_data = final_data.merge(pred, how='left', left_index=True, right_index=True)

_ = final_data[['Price', 'LSTM']].plot(figsize=(10, 4))

# Only predictions
Testing_Period = pd.DataFrame()
Testing_Period['Price'] = sc_inv_y
Testing_Period['LSTM'] = sc_inv_yhat
Testing_Period.index = data.iloc[-len(yhat):,0].index
_ = Testing_Period[['Price', 'LSTM']].plot(figsize=(10, 4))

# Export predictions
Testing_Period.to_csv(r'Data\LSTM_Pred.csv')



# NO SENTIMENT

# load dataset
data = pd.read_csv('Data/All_Merged.csv')
data.isna().sum()
# Inserting 0 for NA
data.fillna(0, inplace=True)
data = data.set_index('date')
data = data.iloc[:,:3]
values = data.values

# Differencing the data. All series become stationary once they have been differenced.
diff_data = data.diff()
values = diff_data.values

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
lags = 1
reframed = series_to_supervised(scaled, lags, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = 2901
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# Som i boosting - men uden validation, kun med rolling window
pred_day = 2901  # Predict for this day, for the next H-1 days. Note indexing of days start from 0.

H = 30  # Forecast horizon, in days. Note there are about 252 trading days in a year
train_size = int(365 * 0.75)  # Use 3 years of data as train set. Note there are about 252 trading days in a year
val_size = int(365 * 0.25)
train_val_size = train_size + val_size  # Size of train+validation set
print("No. of days in train+validation set = " + str(train_val_size))


# Grid search experiment
from itertools import product
pred_day = 2901
neuron = [2, 4, 6]
batch = [7, 14]
epoch = [100, 200]
dropout = [0, 0.2]
parameters = product(neuron, dropout, epoch, batch)
parameters_list = list(parameters)
len(parameters_list)

losses = pd.DataFrame()
j = 1
pred = pd.DataFrame()
while data.index[pred_day] < data.index[len(data) - 1]:
    # NEW: split into input and outputs
    train_X, train_y = values[pred_day - train_val_size:pred_day - val_size, :-1], \
                       values[pred_day - train_val_size:pred_day - val_size, -1]
    val_X, val_y = values[pred_day - val_size:pred_day, :-1], \
                   values[pred_day - val_size:pred_day, -1]
    train_val_X, train_val_y = values[pred_day - train_val_size:pred_day, :-1], \
                               values[pred_day - train_val_size:pred_day, -1]
    test_X, test_y = values[pred_day:pred_day + H, :-1], \
                     values[pred_day:pred_day + H, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
    train_val_X = train_val_X.reshape((train_val_X.shape[0], 1, train_val_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    i = 1
    best_loss = float("inf")
    for param in parameters_list:
        # hidden_node = int(2/3 * train_X.shape[1] * train_X.shape[2])
        model = Sequential()
        model.add(LSTM(param[0], input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(param[1]))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        # # fit network
        history = model.fit(train_X, train_y, epochs=param[2], batch_size=param[3], validation_data=(val_X, val_y), verbose=0,
                         shuffle=False)
        # losses = losses.append(history.history['val_loss'])
        name = str(j) + ',' + str(param)
        # losses[name] = history.history['val_loss']
        loss = history.history['val_loss'][-1]

        print ("Percent finished with grid search for current time prediction:")
        print(round(i/len(parameters_list),2))
        i = i + 1
        if loss < best_loss:
            best_loss = loss
            best_param = param
            print("New best parameters:")
            print(best_param)
            print("_______")

    # Fitting the chosen model
    model = Sequential()
    model.add(LSTM(best_param[0], input_shape=(train_val_X.shape[1], train_val_X.shape[2])))
    model.add(Dropout(best_param[1]))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # # fit network
    history = model.fit(train_val_X, train_val_y, epochs=best_param[2], batch_size=best_param[3], validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # predict using best model grid searched model to test data
    append = pd.DataFrame(model.predict(test_X))
    pred = pd.concat([pred, append], ignore_index=True)

    j = j+1
    print(j)
    pred_day = pred_day + H


# make a prediction
yhat = pred
test_X = scaled[-len(yhat):,1:]  # test[:, :-1]
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X), axis=1)  # [:, 1:]
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert differencing for forecast
actual = Series(data['price'].tail(len(inv_yhat) + 1)).iloc[:-1]
sc_inv_yhat = inv_yhat + actual

# invert scaling for actual
test_y = pd.DataFrame(scaled[-len(yhat):,0])  # pd.DataFrame(test[:, -1])
# test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X), axis=1) # [:, 1:]
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# Invert differencing for actual
sc_inv_y = inv_y + actual

# calculate RMSE
rmse = sqrt(mean_squared_error(sc_inv_y, sc_inv_yhat))
print('Test RMSE: %.3f' % rmse)

final_data = pd.DataFrame()
pred = pd.DataFrame()
pred['LSTM'] = sc_inv_yhat
pred.index = data.iloc[-len(yhat):,0].index
final_data['Price'] = data['price']
final_data = final_data.merge(pred, how='left', left_index=True, right_index=True)

_ = final_data[['Price', 'LSTM']].plot(figsize=(10, 4))

# Only predictions
Testing_Period = pd.DataFrame()
Testing_Period['Price'] = sc_inv_y
Testing_Period['LSTM'] = sc_inv_yhat
Testing_Period.index = data.iloc[-len(yhat):,0].index
_ = Testing_Period[['Price', 'LSTM']].plot(figsize=(10, 4))

# Export predictions
Testing_Period.to_csv(r'Data\LSTM_Pred_NoSent.csv')





# LSTM as forecast combination

# load dataset
data = pd.read_csv('Data/All_Forecasts.csv')
data.isna().sum()
# Inserting 0 for NA
data.fillna(0, inplace=True)
data = data.set_index('date')
price = data[['Price']]
data = data.drop(['AR1'], axis=1)
#data = data.merge(price[['Price']], how='left', left_index=True, right_index=True)
values = data.values

# Differencing the data. All series become stationary once they have been differenced.
diff_data = data.diff()
values = diff_data.values

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
lags = 1
reframed = series_to_supervised(scaled, lags, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[0,10,11,12,13,14,15,16,17]], axis=1, inplace=True)
print(reframed.head())
pred_day =2
j=1
pred = pd.DataFrame()
while data.index[pred_day] < data.index[len(data) - 1]:
    # NEW: split into input and outputs
    train_X, train_y = values[1:pred_day, :-1], \
                       values[1:pred_day, -1]
    test_X, test_y = values[pred_day:pred_day + 1, :-1], \
                     values[pred_day:pred_day + 1, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    i = 1
    # best_loss = float("inf")
    # for param in parameters_list:
    #     # hidden_node = int(2/3 * train_X.shape[1] * train_X.shape[2])
    #     model = Sequential()
    #     model.add(LSTM(param[0], input_shape=(train_X.shape[1], train_X.shape[2])))
    #     model.add(Dropout(param[1]))
    #     model.add(Dense(1))
    #     model.compile(loss='mae', optimizer='adam')
    #
    #     # # fit network
    #     history = model.fit(train_X, train_y, epochs=param[2], batch_size=param[3], verbose=0,
    #                      shuffle=False)
    #
    #     loss = history.history['loss'][-1]
    #
    #     print ("Percent finished with grid search for current time prediction:")
    #     print(round(i/len(parameters_list),2))
    #     i = i + 1
    #     if loss < best_loss:
    #         best_loss = loss
    #         best_param = param
    #         print("New best parameters:")
    #         print(best_param)
    #         print("_______")

    # Fitting the chosen model
    model = Sequential()
    model.add(LSTM(2, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # # fit network
    history = model.fit(train_X, train_y, epochs=10, batch_size=14, validation_data=(test_X, test_y), verbose=0,
                        shuffle=False)
    # predict using best model grid searched model to test data
    append = pd.DataFrame(model.predict(test_X))
    pred = pd.concat([pred, append], ignore_index=True)

    j = j+1
    print(j)
    pred_day = pred_day + 1


# make a prediction
yhat = pred
test_X = scaled[-len(yhat):,1:]  # test[:, :-1]
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X), axis=1)  # [:, 1:]
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert differencing for forecast
actual = Series(data['Price'].tail(len(inv_yhat) + 1)).iloc[:-1]
sc_inv_yhat = inv_yhat + actual

# invert scaling for actual
test_y = pd.DataFrame(scaled[-len(yhat):,0])  # pd.DataFrame(test[:, -1])
# test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X), axis=1) # [:, 1:]
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# Invert differencing for actual
sc_inv_y = inv_y + actual

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

final_data = pd.DataFrame()
pred = pd.DataFrame()
pred['LSTM'] = sc_inv_yhat
pred.index = data.iloc[-len(yhat):,0].index
final_data['Price'] = data['price']
final_data = final_data.merge(pred, how='left', left_index=True, right_index=True)

_ = final_data[['Price', 'LSTM']].plot(figsize=(10, 4))

# Only predictions
Testing_Period = pd.DataFrame()
Testing_Period['Price'] = sc_inv_y
Testing_Period['LSTM'] = sc_inv_yhat
Testing_Period.index = data.iloc[-len(yhat):,0].index
_ = Testing_Period[['Price', 'LSTM']].plot(figsize=(10, 4))

# Export predictions
Testing_Period.to_csv(r'Data\LSTM_Pred_NoSent.csv')

