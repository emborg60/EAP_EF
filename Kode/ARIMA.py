import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import statsmodels
from matplotlib import pyplot
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from pandas import DataFrame
from pandas import concat
from pandas import Series
from math import sqrt
from sklearn.metrics import mean_squared_error

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


data = pd.read_csv('Data/All_Merged.csv')  # , parse_dates=[0], date_parser=dateparse
data.isna().sum()
# Inserting 0 for NA
data.fillna(0, inplace=True)

# plt.figure(figsize=[10,4])
# plt.title('BTC Price (USD) Daily')
# plt.plot(data.price, '-', label='Daily')

# Monthly
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].dt.tz_localize(None)
data = data.groupby([pd.Grouper(key='date', freq='M')]).first().reset_index()
data = data.set_index('date')
data['price'].fillna(method='ffill', inplace=True)

# Decomposition - only for price though!
# decomposition = sm.tsa.seasonal_decompose(data.price)
#
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid
#
# fig = plt.figure(figsize=(10,8))
#
# plt.subplot(411)
# plt.plot(data.price, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='best')
#
# fig.suptitle('Decomposition of Prices Data')
# plt.show()

# Setting the data structure
reframed = series_to_supervised(data, 1, 1)
# Also removing the lagged price, as this will be created in the ARIMA
reframed.drop(reframed.columns[[0,8, 9, 10, 11, 12, 13]], axis=1, inplace=True)
print(reframed.head())

# split data
split_date = '2018-06-25'
reframed_train = reframed.loc[reframed.index <= split_date].copy()
reframed_test = reframed.loc[reframed.index > split_date].copy()

# Prøver lige ARIMA på original data
# Det her er en seasonal ARIMA, SARIMA, så nok ekstra resultat efter en regulær ARIMA
# Hjælp til kommentering findes her: https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/
# Den fitter fint hvis man ikke opdeler i train og test..
# Initial approximation of parameters
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

x_train = reframed_train.iloc[:,:-1].values
y_train = reframed_train.iloc[:,-1]
x_test = reframed_test.iloc[:,:-1].values
y_test = reframed_test.iloc[:,-1]


# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(endog=y_train, exog=x_train, order=(param[0], d, param[1]),
                                        seasonal_order=(param[2], D, param[3], 12),enforce_stationarity=True,
                                            enforce_invertibility=True).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())

# Residual plot of the best model
fig = plt.figure(figsize=(10,4))
best_model.resid.plot()
fig.suptitle('Residual Plot of the Best Model')
print("Dickey–Fuller test:: p=%f" % sm.tsa.stattools.adfuller(best_model.resid)[1])
# Dickey–Fuller test:: p=0.xxx -> Residuals are stationary


df_month2 = data[['price']]
future = pd.DataFrame()
df_month2 = pd.concat([df_month2, future])
df_month2['forecast'] = best_model.predict(start = len(x_train), end = len(x_train)+len(x_test)-1, exog=x_test)
plt.figure(figsize=(8,4))
df_month2.price.plot()
df_month2.forecast.plot(color='r', ls='--', label='Predicted Price')
plt.legend()
plt.title('Bitcoin Prices (USD) Predicted vs Actuals, by months')
plt.ylabel('mean USD')
plt.show()


# Daily version
df = pd.read_csv('Data/All_Merged.csv')
df.isna().sum()
# Inserting 0 for NA
df.fillna(0, inplace=True)

# Date type
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.tz_localize(None)
df = df.groupby([pd.Grouper(key='date', freq='D')]).first().reset_index()
df = df.set_index('date')
df['price'].fillna(method='ffill', inplace=True)


# Setting the data structure
daily_re = series_to_supervised(df, 1, 1)
price = daily_re.iloc[:,0]
da_price = daily_re.iloc[:,0]
daily_re.drop(daily_re.columns[[0,8, 9, 10, 11, 12, 13]], axis=1, inplace=True)
y = daily_re.iloc[:,-1]
print(daily_re.head())

# split data
split_date = '2018-07-11'
daily_re_train = daily_re.loc[daily_re.index <= split_date].copy()
daily_re_test = daily_re.loc[daily_re.index > split_date].copy()
da_price = da_price.loc[da_price.index > split_date].copy()
da_price = da_price.values

# Som i boosting - men uden validation, kun med rolling window
pred_day = 2901-16 # Predict for this day, for the next H-1 days. Note indexing of days start from 0.

H = 30  # Forecast horizon, in days. Note there are about 252 trading days in a year
train_size = int(365 * 0.75)  # Use 3 years of data as train set. Note there are about 252 trading days in a year
val_size = int(365 * 0.25)

train_val_size = train_size + val_size  # Size of train+validation set
print("No. of days in train+validation set = " + str(train_val_size))

qs = range(0, 3)
ps = range(1, 3)
d = 1
parameters = product(ps, qs)
parameters_list = list(parameters)
len(parameters_list)

# ARIMA igen, men ikke som seasonal og med Training og Test data, på daily niveau
# Initial approximation of parameters
pred = pd.DataFrame()
while daily_re.index[pred_day] < daily_re.index[len(daily_re) - 1]:

    x_da_train = daily_re.iloc[pred_day - train_val_size:pred_day,:-1].values
    y_da_train = daily_re.iloc[pred_day - train_val_size:pred_day,-1]
    x_da_test = daily_re.iloc[pred_day:pred_day + H,:-1].values
    y_da_test = daily_re.iloc[pred_day:pred_day + H,-1]

    # Model Selection
    results = []
    best_bic = float("inf")
    warnings.filterwarnings('ignore')
    for param in parameters_list:
        try:
            model=sm.tsa.ARIMA(endog=y_da_train, exog=x_da_train, order=(param[0], d, param[1])).fit(disp=-1)
        except ValueError:
            print('wrong parameters:', param)
            continue
        bic = model.bic
        if bic < best_bic:
            best_model = model
            best_bic = bic
            best_param = param
        results.append([param, model.aic])

    # Best Models
# result_table = pd.DataFrame(results)
# result_table.columns = ['parameters', 'bic']
# print(result_table.sort_values(by = 'bic', ascending=True).head())
# print(best_model.summary())
    append = best_model.predict(start = len(x_da_train), end = len(x_da_train)+len(x_da_test)-1, exog=x_da_test).T
    pred = pd.concat([pred, append], ignore_index=True)

    pred_day = pred_day + H

pred_day = 2901-16  # Reset
price2 = price.iloc[pred_day:]
pred['prev_price'] = price2.values
pred.index = price2.index
pred['pred'] = pred.sum(axis=1)

# price2 = price2.values
# pred = pred.values


# Residual plot of the best model
# fig = plt.figure(figsize=(10,4))
# best_model.resid.plot()
# fig.suptitle('Residual Plot of the Best Model')
# print("Dickey–Fuller test:: p=%f" % sm.tsa.stattools.adfuller(best_model.resid)[1])
# Dickey–Fuller test:: p=0.001213 -> Residuals are stationary


df_month2 = df[['price']]
future = pd.DataFrame()
df_month2 = pd.concat([df_month2, future])
yhat = (pred.T + price2).T.astype('float32')
df_month2['forecast'] = pred['pred']
plt.figure(figsize=(8,4))
df_month2.price.plot()
df_month2.forecast.plot(color='r', ls='--', label='Predicted Price')
plt.legend()
plt.title('Bitcoin Prices (USD) Predicted vs Actuals, by months')
plt.ylabel('mean USD')
plt.show()

# RMSE
y = y.iloc[pred_day:]
yhat = pred['pred']
rmse = sqrt(mean_squared_error(y, yhat))
print('Test RMSE: %.3f' % rmse)

# Only the forecast part
da_fore = daily_re_test[['var1(t)']]
future = pd.DataFrame()
da_fore = pd.concat([da_fore, future])
da_fore['forecast'] = pred['pred']
plt.figure(figsize=(8,4))
da_fore['var1(t)'].plot()
da_fore.forecast.plot(color='r', ls='--', label='Predicted Price')
plt.legend()
plt.title('Bitcoin Prices (USD) Predicted vs Actuals, by months')
plt.ylabel('mean USD')
plt.show()

# Export predictions
pred['pred'].to_csv(r'Data\ARIMA_Pred.csv')


#### Appendix ####
def descriptive_statistics(df, series):
    stats = df[series].describe()
    print('\nDescriptive Statistics for', '\'' + series + '\'', '\n\n', stats)


def get_graphics(df, series, xlabel, ylabel, title, grid=True):
    plt.plot(pd.to_datetime(df.index), df[series])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)
    return plt.show()


# stationary tests
# unit root = statistical properties of series are not constant with time.
#
# In order to be stationary, series has to be constant with time. So if a series has a unit root, it is not stationary
#
# strict stationary = mean, variance, covariance are not function of time
# trend stationary = no root unit, but has a trend. if you remove the trend, it would be strict stationary
# difference stationary = series can be made strict stationary by differencing

# ADF Augmented Dickey Fuller Test (unit root test)
# null hypothesis = series has a unit root (a = 1)
# alt hypothesis = series has no unit root
#
# accept null = t-score is greter than critical value (there is a unit root)
# reject null = t-score is less than critical value (there is no unit root)
#
# accpet null = bad (not stationary)
# reject null = good (stationary)
#
# adf can be interpreted as a difference stationary test


def adf_test(df, series):
    results = adfuller(df[series])
    output = pd.Series(results[0:4], index=['t-score', 'p-value', '# of lags used', '# of observations'])
    for key, value in results[4].items():
        output['critical value (%s)' % key] = value
    # if t-score < critical value at 5%, the data is stationary
    # if t-score > critical value at 5%, the data is NOT stationary
    if output[0] < output[5]:
        print('\nADF: The data', '\'' + series + '\'', 'is STATIONARY \n\n', output)
    elif output[0] > output[5]:
        print('\nADF: The data', '\'' + series + '\'', 'is NOT STATIONARY \n\n', output)
    else:
        print('\nADF: There is something wrong with', '\'' + series + '\'', '\n\n', output)


# KPSS Kwiatkowski-Phillips-Schmidt-Shin Test (stationary test)
# null hypothesis = the series has a stationary trend
# alt hypothesis = the series has a unit root (series is not stationary)
#
# accept null = t-score is less than critical value (series is stationary)
# reject null = t-score is greater than the critical value (series is not stationary)
#
# accpet null = good (stationary)
# reject null = bad (not stationary)
#
# kpss classifies a series as stationary on the absence of a unit root
# (both strict stationary and trend stationary will be classified as stationary)


def kpss_test(df, series):
    results = kpss(df[series], regression='ct')
    output = pd.Series(results[0:3], index=['t-score', 'p-value', '# lags used'])
    for key, value in results[3].items():
        output['critical value (%s)' % key] = value
    # if t-score < critical value at 5%, the data is stationary
    # if t-score > critical value at 5%, the data is NOT stationary
    if output[0] < output[4]:
        print('\nKPSS: The data', '\'' + series + '\'', 'is STATIONARY \n\n', output)
    elif output[0] > output[4]:
        print('\nKPSS: The data', '\'' + series + '\'', 'is NOT STATIONARY \n\n', output)
    else:
        print('\nKPSS: There is something wrong with', '\'' + series + '\'', '\n\n', output)


# Many times, adf and kpss can give conflicting results. if so:
#
# [adf = stationary], [kpss = stationary] = series is stationary
# [adf = stationary], [kpss = NOT stationary] = series is difference stationary. use differencing to make it stationary
# [adf = NOT stationary], [kpss = stationary] = series is trend stationary. remove trend to make strict stationary
# [adf = NOT STATIONARY], [kpss = NOT STATIONARY] = series is not stationary


def series_analysis(df, series, xlabel, ylabel, title):
    # descriptive stats
    descriptive_statistics(df, series)
    # graphics
    get_graphics(df, series, xlabel, ylabel, title, grid = True)
    # stationary tests
    adf_test(df, series)
    kpss_test(df, series)


# Så går vi i gang:

# create new df for stationary data
stationary = pd.DataFrame()

# ['price']
series_analysis(df, 'price', xlabel = 'year', ylabel = 'Bitcoin Price(USD)', title = 'df[\'price\']')
# ADF: "Not Stationary"
# KPSS: "Not Stationary"

# ['price'] =  diff
stationary['price'] = df['price'].diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'price', xlabel = 'year', ylabel = 'Bitcoin Price(USD)', title = 'Log_Diff_Price')
# ADF: "Stationary"
# KPSS: "Stationary"

# ['volatility']
series_analysis(df, 'volatility', xlabel = 'year', ylabel = 'volatility (daily (high-low)/price)', title = 'df[\'volatility\']')
# ADF: Stationary
# KPSS: Not Stationary

# ['volatility'] = diff
stationary['volatility'] = df['volatility'].diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'volatility',  xlabel = 'year', ylabel = 'Volatility', title = 'Diff_Volatility')
# ADF: "Stationary"
# KPSS: "Stationary"

# ['volume_price']
series_analysis(df, 'volume_price', xlabel = 'year', ylabel = 'Volume_Price(USD)', title = 'df[\'volume_price\']')
# ADF: "Not Stationary"
# KPSS: "Not Stationary"

# ['volume_price'] = log & diff
stationary['volume_price'] =  df['volume_price'].diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'volume_price',  xlabel = 'year', ylabel = 'volume_price', title = 'Log_Diff_Volume_Price')
# ADF: "Stationary"
# KPSS: "Stationary"

# ['volume_number']
series_analysis(df, 'volume_number', xlabel = 'year', ylabel = 'Number of Bitocins exchanged', title = 'df[\'volume_number\']')
# ADF: "Not Stationary"
# KPSS: "Not Stationary"

# ['volume_number'] = log & diff
stationary['volume_number'] = df['volume_number'].diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'volume_number',  xlabel = 'year', ylabel = 'volume_number', title = 'Diff_Volume_Number')
# ADF: "Stationary"
# KPSS: "Stationary"

# ['positive_comment']
series_analysis(df, 'positive_comment', xlabel = 'year', ylabel = 'Number of Positive Comments', title = 'df[\'positive_comment\']')
# ADF: "Stationary"
# KPSS: "Not Stationary"

# ['positive_comment'] = diff
stationary['positive_comment'] = df['positive_comment'].diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'positive_comment', xlabel = 'year', ylabel = 'Number of Positive Comments', title = 'Diff_Positive_Comment')
# ADF: "Stationary"
# KPSS: "Stationary"


# ['neutral_comment']
series_analysis(df, 'neutral_comment', xlabel = 'year', ylabel = 'Number of Neutral Comments', title = 'df[\'neutral_comment\']')
# ADF: "Stationary"
# KPSS: "Not Stationary"

# ['neutral_comment'] = diff
stationary['neutral_comment'] = df['neutral_comment'].diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'neutral_comment', xlabel = 'year', ylabel = 'Number of Neutral Comments', title = 'Diff_Neutral_Comment')
# ADF: "Stationary"
# KPSS: "Stationary"


# ['negative_comment']
series_analysis(df, 'negative_comment', xlabel = 'year', ylabel = 'Number of Negative Comments', title = 'df[\'negative_comment\']')
# ADF: "Stationary"
# KPSS: "Not Stationary"

# ['negative_comment'] = diff
stationary['negative_comment'] = df['negative_comment'].diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'negative_comment', xlabel = 'year', ylabel = 'Number of Negative Comments', title = 'Diff_Negative_Comment')
# ADF: "Stationary"
# KPSS: "Stationary"


