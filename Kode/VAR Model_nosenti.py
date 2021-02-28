# import packages
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from pandas import Series
from math import sqrt
from sklearn.metrics import mean_squared_error

# import data
df = pd.read_csv('XGB_seb/Data/data_daily.csv')
del df['Unnamed: 0']
del df['positive_comment'],
del df['neutral_comment']
del df['negative_comment']

df_stats = pd.DataFrame(df.mean().T)
df_stats['Std'] = pd.DataFrame(df.std().T)
df_stats['Skewness'] = pd.DataFrame(df.skew().T)
df_stats['Kurtosis'] = pd.DataFrame(df.kurt().T)
df_stats.columns = ['Mean','Std','Skewness','Kurtosis']

df_stats = round(df_stats,2)
print(round(df_stats,2).to_latex())

df = df.fillna(0)
df.index = pd.to_datetime(df['date'])
dflevel = df.copy()


df = df.fillna(0)
dflevel = df.copy()

features = [
    'price',
    'volatility',
    'volume_price',
    'volume_number',
]

for feat in features:
    df[feat] = df[feat].diff()

df = df.dropna()

pred_day = 2901  # Predict for this day, for the next H-1 days. Note indexing of days start from 0.

H = 30  # Forecast horizon, in days.

train_val_size = int(365 * 1)  # Size of train+validation set

# stationary index to datetime
# stationary.index = pd.DatetimeIndex(stationary.index).date
df.index = pd.to_datetime(df['date'])

# For at undgå "ValueWarning" ved "model = VAR(endog = train)"
# stationary['date'] = pd.to_datetime(stationary['date'])
# stationary['date'] = stationary['date'].dt.tz_localize(None)
# stationary = stationary.groupby([pd.Grouper(key='date', freq='D')]).first().reset_index()
# stationary = stationary.set_index('date')
# stationary['price'].fillna(method='ffill', inplace=True)


# pick start date for your analysis
# look at last step and see when data starts to be consistent
## Det skal vi lige se på det her
# start_date = dt.date(2012, 1, 1)
# stationary = stationary[start_date : ]
# stationary.head()

# create train data
H = 30
all_fc_level = pd.DataFrame()
while df.iloc[pred_day, 0] < df.iloc[len(df) - 1, 0]:
    print("Predicting on day %d, date %s, with forecast horizon H = %d" % (pred_day, df.iloc[pred_day]['date'], H))

    sfromdate = df.iloc[pred_day, 0]
    stodate = df.iloc[pred_day + H, 0]

    train_val = df[pred_day - train_val_size:pred_day].copy()
    test = df[pred_day:pred_day + H].copy()

    # create VAR model on train data
    model = VAR(endog=train_val[features])

    lag = model.select_order(maxlags=10, trend='c').selected_orders['bic']
    print(lag)
    model_fit = model.fit(lag, ic='bic', trend='c')


    for i in range(30):
        fc = model_fit.forecast(y=df[pred_day - lag + i: pred_day + i][features].values, steps=1)
        # revert difference. add 1 lagged true values to fc
        fc_lvl = fc + dflevel[pred_day + i: pred_day + i +1][features].values
        fc_lvl = pd.DataFrame(fc_lvl, index=[test.iloc[i, 0]], columns=features)

        all_fc_level = pd.concat([all_fc_level, fc_lvl])

    pred_day = pred_day + H

all_fc_level = all_fc_level.join(dflevel['price'], rsuffix='_true')
all_fc_level.to_csv(r'Data\VAR_pred_nosenti.csv')

rmse(all_fc_level['price'],all_fc_level['price_true'])

