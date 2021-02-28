import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Data/All_Merged.csv')  # , parse_dates=[0], date_parser=dateparse
data.isna().sum()
# Inserting 0 for NA
data.fillna(0, inplace=True)


# LSTM
data = data.set_index('date')
data = data['price']
# split data
split_date = '2018-06-25'
data_train = data.loc[data.index <= split_date].copy()
data_test = data.loc[data.index > split_date].copy()

# Preprocessing of data
training_set = data_train.values
training_set = np.reshape(training_set, (len(training_set), 1))


sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set) - 1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = data.plot(style='', figsize=(15, 5), color=color_pal[0], title='BTC Price (USD) Daily')

# _ = data_test \
#    .rename(columns={'price': 'Test Set'}) \
#    .join(data_train.rename(columns={'price': 'Training Set'}), how='outer') \
#    .plot(figsize=(15,5), title='BTC Weighted_Price Price (USD) by Hours', style='')

# Vanilla LSTM
# Importing the Keras libraries and packages


model = Sequential()
model.add(LSTM(128, activation="sigmoid", input_shape=(1, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=2)

# Making the predictions
test_set = data_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = model.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)

datatest = pd.DataFrame()
datatest['Test Prices'] = data_test
datatest['Price_Prediction'] = predicted_BTC_price
# data_test['Price_Prediction'] = predicted_BTC_price
# data_all = pd.concat([data_test, data_train], sort=False)

# saving the predicted values in a common data frame for future comparision
# final_data = data_all
# final_data = final_data.reset_index()
# final_data = final_data.rename(columns={'price': 'lstm'})
# final_data = final_data[['date','lstm']]

_ = datatest[['Test Prices', 'Price_Prediction']].plot(figsize=(15, 5))

# calculate MSE and MAE

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse1 = mean_squared_error(y_true=datatest['Test Prices'],
                   y_pred=datatest['Price_Prediction'])

mae1 = mean_absolute_error(y_true=datatest['Test Prices'],
                    y_pred=datatest['Price_Prediction'])

# LSTM with more variables
data = pd.read_csv('Data/All_Merged.csv')  # , parse_dates=[0], date_parser=dateparse
data.isna().sum()
# Inserting 0 for NA
data.fillna(0, inplace=True)

data = data.set_index('date')
# split data
split_date = '2018-25-06'
data_train = data.loc[data.index <= split_date].copy()
data_test = data.loc[data.index > split_date].copy()

# Preprocessing of data
training_set = data_train.values
training_set = np.reshape(training_set, (len(training_set), 7))
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set) - 1]
y_train = training_set[1:len(training_set)][:, 0]
y_train = np.reshape(y_train, (y_train.shape[0], 1))
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

model = Sequential()
model.add(LSTM(100, activation="sigmoid", input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=500, batch_size=50, verbose=2)

# Predictions
test_set = data_test.values
inputs = np.reshape(test_set, (len(test_set), 7))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 7))
predicted_BTC_price = model.predict(inputs)
for i in range(1, 4):
    predicted_BTC_price = np.append(predicted_BTC_price, predicted_BTC_price, axis=1)
predicted_BTC_price = predicted_BTC_price[:, 0:7]
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)
predicted_BTC_price = predicted_BTC_price[:, 0]

datatest = pd.DataFrame()
datatest['Test Prices'] = data_test['price']
datatest['Price_Prediction'] = predicted_BTC_price

_ = datatest[['Test Prices', 'Price_Prediction']].plot(figsize=(15, 5))

# calculate MSE and MAE

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse2 = mean_squared_error(y_true=datatest['Test Prices'],
                   y_pred=datatest['Price_Prediction'])

mae2 = mean_absolute_error(y_true=datatest['Test Prices'],
                    y_pred=datatest['Price_Prediction'])

# Den er dårligere end bare prisen?? Noget må være galt
