import chart_studio.plotly as py
import chart_studio
import math
import matplotlib
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objs as go
import time

from os import listdir
from os.path import isfile, join


from collections import defaultdict
from fastai.tabular import add_datepart
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

chart_studio.tools.set_credentials_file(username='Emborg', api_key='bbBdW78XyA7bPc9shlkf')


def train_pred_eval_model(X_train,
                          y_train,
                          X_test,
                          y_test,
                          seed=100,
                          n_estimators=100,
                          max_depth=3,
                          learning_rate=0.1,
                          min_child_weight=1,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          gamma=0):
    '''
    Train model, do prediction, scale back to original range and do evaluation
    Use XGBoost here.
    Inputs
        X_test            : features of the test set
        y_test             : target for test. Actual values, not scaled.
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :
    Outputs
        rmse               : root mean square error of y_test and est
        mape               : mean absolute percentage error of y_test and est
        mae                : mean absolute error of y_test and est
        est                : predicted values. Same length as y_test
    '''

    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         gamma=gamma,
                         seed=seed,
                         booster='gbtree')

    # Train the model
    model.fit(X_train, y_train)

    # Get predicted labels and scale back to original range

    est = model.predict(X_test)

    # Calculate RMSE, MAPE, MAE
    rmse = get_rmse(y_test, est)
    mape = get_mape(y_test, est)
    mae = get_mae(y_test, est)

    return rmse, mape, mae, est, model.feature_importances_


def add_lags(df, N, lag_cols):
    """
    Add lags up to N number of days to use as features
    The lag columns are labelled as 'price_lag_1', 'price_lag_2', ... etc.
    """
    # Use lags up to N number of days to use as features
    df_w_lags = df.copy()
    df_w_lags.loc[:, 'order_day'] = [x for x in list(
        range(len(df)))]  # Add a column 'order_day' to indicate the order of the rows by date
    merging_keys = ['order_day']  # merging_keys
    shift_range = [x + 1 for x in range(N)]
    for shift in shift_range:
        train_shift = df_w_lags[merging_keys + lag_cols].copy()

        # E.g. order_day of 0 becomes 1, for shift = 1.
        # So when this is merged with order_day of 1 in df_w_lags, this will represent lag of 1.
        train_shift['order_day'] = train_shift['order_day'] + shift

        foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
        train_shift = train_shift.rename(columns=foo)

        df_w_lags = pd.merge(df_w_lags, train_shift, on=merging_keys, how='left')  # .fillna(0)
    del train_shift

    return df_w_lags


def get_error_metrics(df,
                      train_size,
                      N,
                      H,
                      seed=100,
                      n_estimators=100,
                      max_depth=3,
                      learning_rate=0.1,
                      min_child_weight=1,
                      subsample=1,
                      colsample_bytree=1,
                      colsample_bylevel=1,
                      gamma=0):
    """
    Given a series consisting of both train+validation, do predictions of forecast horizon H on the validation set,
    at H/2 intervals.
    Inputs
        df                 : train + val dataframe. len(df) = train_size + val_size
        train_size         : size of train set
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              : learning rate

    Outputs
        mean of rmse, mean of mape, mean of mae, dictionary of predictions
    """
    rmse_list = []  # root mean square error
    mape_list = []  # mean absolute percentage error
    mae_list = []  # mean absolute error
    preds_dict = {}

    # Add lags up to N number of days to use as features
    df = add_lags(df, N, ['price'])
    df = add_lags(df, N, ['volatility'])
    df = add_lags(df, N, ['volume_price'])
    df = add_lags(df, N, ['volume_number'])


    # Get list of features
    features = [
        'year',
        'month',
        'week',
        'day',
        'dayofweek',
        'dayofyear',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end'
    ]

    for n in range(N, 0, -1):
        features.append("volatility_lag_" + str(n))
        features.append("volume_number_lag_" + str(n))
        features.append("volume_price_lag_" + str(n))
        features.append("price_lag_" + str(n))

    # features contain all features, including price_lags

    for i in range(train_size, len(df) - H + 1, int(H / 2)):
        # Split into train and test
        train = df[i - train_size:i].copy()
        test = df[i:i + H].copy()

        # Drop the NaNs in train
        train.dropna(axis=0, how='any', inplace=True)

        # Split into X and y
        X_train = train[features]
        y_train = train['price']
        X_test = test[features]
        y_test = test['price']


        rmse, mape, mae, est, _ = train_pred_eval_model(X_train,
                                                        y_train,
                                                        X_test,
                                                        y_test,
                                                        seed=seed,
                                                        n_estimators=n_estimators,
                                                        max_depth=max_depth,
                                                        learning_rate=learning_rate,
                                                        min_child_weight=min_child_weight,
                                                        subsample=subsample,
                                                        colsample_bytree=colsample_bytree,
                                                        colsample_bylevel=colsample_bylevel,
                                                        gamma=gamma)

        rmse_list.append(rmse)
        mape_list.append(mape)
        mae_list.append(mae)
        preds_dict[i] = est

    return np.mean(rmse_list), np.mean(mape_list), np.mean(mae_list), preds_dict


def get_error_metrics_one_pred(df,
                               train_size,
                               N,
                               H,
                               seed=100,
                               n_estimators=100,
                               max_depth=3,
                               learning_rate=0.1,
                               min_child_weight=1,
                               subsample=1,
                               colsample_bytree=1,
                               colsample_bylevel=1,
                               gamma=0):
    """
    Given a series consisting of both train+test, do one prediction of forecast horizon H on the test set.
    Inputs
        df                 : train + test dataframe. len(df) = train_size + test_size
        train_size         : size of train set
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :

    Outputs
        rmse, mape, mae, predictions
    """

    # Add lags up to N number of days to use as features
    df = add_lags(df, N, ['price'])
    df = add_lags(df, N, ['volatility'])
    df = add_lags(df, N, ['volume_price'])
    df = add_lags(df, N, ['volume_number'])

    # Get list of features
    features = [
        'year',
        'month',
        'week',
        'day',
        'dayofweek',
        'dayofyear',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end'
    ]

    for n in range(N, 0, -1):
        features.append("volatility_lag_" + str(n))
        features.append("volume_number_lag_" + str(n))
        features.append("volume_price_lag_" + str(n))
        features.append("price_lag_" + str(n))

    # features contain all features, including price_lags

    # Split into train and test
    train = df[:train_size].copy()
    test = df[train_size:train_size + H].copy()

    # Drop the NaNs in train
    train.dropna(axis=0, how='any', inplace=True)

    # Split into X and y
    X_test = test[features]
    y_test = test['price']
    X_train = train[features]
    y_train = train['price']

    rmse, mape, mae, est, feature_importances = train_pred_eval_model(X_test,
                                                                      y_test,
                                                                      X_train,
                                                                      y_train,
                                                                      seed=seed,
                                                                      n_estimators=n_estimators,
                                                                      max_depth=max_depth,
                                                                      learning_rate=learning_rate,
                                                                      min_child_weight=min_child_weight,
                                                                      subsample=subsample,
                                                                      colsample_bytree=colsample_bytree,
                                                                      colsample_bylevel=colsample_bylevel,
                                                                      gamma=gamma)

    return rmse, mape, mae, est, feature_importances, features


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_mae(a, b):
    """
    Comp mean absolute error e_t = E[|a_t - b_t|]. a and b can be lists.
    Returns a vector of len = len(a) = len(b)
    """
    return np.mean(abs(np.array(a) - np.array(b)))


def get_rmse(a, b):
    """
    Comp RMSE. a and b can be lists.
    Returns a scalar.
    """
    return math.sqrt(np.mean((np.array(a) - np.array(b)) ** 2))


tic1 = time.time()

#### Load data:

df = pd.read_csv('XGB_seb/Data/data_daily.csv')
del df['Unnamed: 0']
df = df.fillna(0)

dflevel = df.copy()

features = [
    'price',
    'volatility',
    'volume_price',
    'volume_number',
    'positive_comment',
    'neutral_comment',
    'negative_comment'
]

for feat in features:
    df[feat] = df[feat].diff()

df = df.dropna()

pred_day = 2901  # Predict for this day, for the next H-1 days. Note indexing of days start from 0.

H = 30  # Forecast horizon, in days. Note there are about 252 trading days in a year
train_size = int(365 * 0.75)  # Use 3 years of data as train set. Note there are about 252 trading days in a year
val_size = int(365 * 0.25)  # Use 1 year of data as validation set
N = 1  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features

n_estimators = 100  # Number of boosted trees to fit. default = 100
max_depth = 3  # Maximum tree depth for base learners. default = 3
learning_rate = 0.1  # Boosting learning rate (xgb’s “eta”). default = 0.1
min_child_weight = 1  # Minimum sum of instance weight(hessian) needed in a child. default = 1
subsample = 1  # Subsample ratio of the training instance. default = 1
colsample_bytree = 1  # Subsample ratio of columns when constructing each tree. default = 1
colsample_bylevel = 1  # Subsample ratio of columns for each split, in each level. default = 1
gamma = 0  # Minimum loss reduction required to make a further partition on a leaf node of the tree. default=0

model_seed = 100

fontsize = 14
ticklabelsize = 14

# Plotly colors
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]

train_val_size = train_size + val_size  # Size of train+validation set
print("No. of days in train+validation set = " + str(train_val_size))

### Feature engineering:
add_datepart(df, 'date', drop=False)
df.drop('Elapsed', axis=1, inplace=True)  # don't need this
df.head(50)

df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

df.loc[:, 'year'] = LabelEncoder().fit_transform(df['year'])
df[15:25]
all_fc = pd.DataFrame()
df.isnull().sum()
# Start rolling fc
while df.iloc[pred_day, 0] < df.iloc[len(df) - 1, 0]:
    print("Predicting on day %d, date %s, with forecast horizon H = %d" % (pred_day, df.iloc[pred_day]['date'], H))

    sfromdate = df.iloc[pred_day, 0]
    stodate = df.iloc[pred_day + H, 0]

    train = df[pred_day - train_val_size:pred_day - val_size].copy()
    val = df[pred_day - val_size:pred_day].copy()
    train_val = df[pred_day - train_val_size:pred_day].copy()
    test = df[pred_day:pred_day + H].copy()

    print("train.shape = " + str(train.shape))
    print("val.shape = " + str(val.shape))
    print("train_val.shape = " + str(train_val.shape))
    print("test.shape = " + str(test.shape))

    # # Get error metrics on validation set before hyperparameter tuning
    rmse_bef_tuning, mape_bef_tuning, mae_bef_tuning, preds_dict = \
        get_error_metrics(train_val,
                          train_size,
                          N,
                          H,
                          seed=model_seed,
                          n_estimators=n_estimators,
                          max_depth=max_depth,
                          learning_rate=learning_rate)
    print("RMSE = %0.3f" % rmse_bef_tuning)
    print("MAPE = %0.3f%%" % mape_bef_tuning)
    print("MAE = %0.3f" % mae_bef_tuning)

    test_rmse_bef_tuning, test_mape_bef_tuning, test_mae_bef_tuning, est, feature_importances, features = \
        get_error_metrics_one_pred(df[pred_day - train_val_size:pred_day + H],
                                   train_size + val_size,
                                   N,
                                   H,
                                   seed=model_seed,
                                   n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate)

    print("RMSE = %0.3f" % test_rmse_bef_tuning)
    print("MAPE = %0.3f%%" % test_mape_bef_tuning)
    print("MAE = %0.3f" % test_mae_bef_tuning)

    # fig = go.Figure()

    # # Add traces
    # fig.add_trace(go.Scatter(x=train['date'],
    #                          y=train['price'],
    #                          mode='lines',
    #                          name='train',
    #                          line=dict(color='blue')))
    # fig.add_trace(go.Scatter(x=val['date'],
    #                          y=val['price'],
    #                          mode='lines',
    #                          name='validation',
    #                          line=dict(color='orange')))
    # fig.add_trace(go.Scatter(x=test['date'],
    #                          y=test['price'],
    #                          mode='lines',
    #                          name='test',
    #                          line=dict(color='green')))
    # fig.add_trace(go.Scatter(x=test[:H]['date'],
    #                          y=est,
    #                          mode='lines',
    #                          name='predictions',
    #                          line=dict(color='red')))
    # fig.update_layout(yaxis=dict(title='USD'),
    #                   xaxis=dict(title='date'))
    #
    # py.plot(fig, filename='price_fc_untuned')
    #
    # imp = list(zip(features, feature_importances))
    # imp.sort(key=lambda tup: tup[1], reverse=False)
    # imp

    # # Plot the importance scores as a bar chart
    # fig = go.Figure(go.Bar(
    #             x=[item[1] for item in imp[-15:]],
    #             y=[item[0] for item in imp[-15:]],
    #             orientation='h'))
    # fig.update_layout(yaxis=dict(title='feature'),
    #                   xaxis=dict(title='relative importance'))
    # py.plot(fig, filename='importance_untuned')


    ################ Tuning n_estimators, max_depht
    param_label = 'n_estimators'
    param_list = [500, 200, 175, 150, 125, 100, 75, 50, 25]

    param2_label = 'max_depth'
    param2_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    error_rate = defaultdict(list)
    N_opt = 1
    tic = time.time()
    for param in param_list:
        for param2 in param2_list:
            print(str(param) + ' & ' + str(param2))
            rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(train_val,
                                                                  train_size,
                                                                  N_opt,
                                                                  H,
                                                                  seed=model_seed,
                                                                  n_estimators=param,
                                                                  max_depth=param2,
                                                                  learning_rate=learning_rate)

            # Collect results
            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate['rmse'].append(rmse_mean)
            error_rate['mape'].append(mape_mean)
            error_rate['mae'].append(mae_mean)

    error_rate = pd.DataFrame(error_rate)
    toc = time.time()
    print("Minutes taken = {0:.2f}".format((toc - tic) / 60.0))

    error_rate

    # Get optimum value for param and param2, using RMSE
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    n_estimators_opt = temp['n_estimators'].values[0]
    max_depth_opt = temp['max_depth'].values[0]
    print("min RMSE = %0.3f" % error_rate['rmse'].min())
    print("optimum params = ")
    n_estimators_opt, max_depth_opt

    ############ TODO Tuning learning_rate & min_child_weight

    param_label = 'learning_rate'
    param_list = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]

    param2_label = 'min_child_weight'
    param2_list = [1]

    error_rate = defaultdict(list)

    tic = time.time()
    for param in param_list:
        for param2 in param2_list:
            rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(train_val,
                                                                  train_size,
                                                                  N_opt,
                                                                  H,
                                                                  seed=model_seed,
                                                                  n_estimators=n_estimators_opt,
                                                                  max_depth=max_depth_opt,
                                                                  learning_rate=param,
                                                                  min_child_weight=param2)

            # Collect results
            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate['rmse'].append(rmse_mean)
            error_rate['mape'].append(mape_mean)
            error_rate['mae'].append(mae_mean)

    error_rate = pd.DataFrame(error_rate)
    toc = time.time()
    print("Minutes taken = {0:.2f}".format((toc - tic) / 60.0))

    error_rate


    # Get optimum value for param and param2, using RMSE
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    learning_rate_opt = temp['learning_rate'].values[0]
    min_child_weight_opt = temp['min_child_weight'].values[0]
    print("min RMSE = %0.3f" % error_rate['rmse'].min())
    print("optimum params = ")
    learning_rate_opt, min_child_weight_opt


    # ############ TODO Tuning subsample & gamma
    #
    # param_label = 'subsample'
    # param_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #
    # param2_label = 'gamma'
    # param2_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #
    # error_rate = defaultdict(list)
    #
    # tic = time.time()
    # for param in param_list:
    #     for param2 in param2_list:
    #         rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(train_val,
    #                                                               train_size,
    #                                                               N_opt,
    #                                                               H,
    #                                                               seed=model_seed,
    #                                                               n_estimators=n_estimators_opt,
    #                                                               max_depth=max_depth_opt,
    #                                                               learning_rate=learning_rate_opt,
    #                                                               min_child_weight=min_child_weight_opt,
    #                                                               subsample=param,
    #                                                               colsample_bytree=colsample_bytree,
    #                                                               colsample_bylevel=colsample_bylevel,
    #                                                               gamma=param2)
    #
    #         # Collect results
    #         error_rate[param_label].append(param)
    #         error_rate[param2_label].append(param2)
    #         error_rate['rmse'].append(rmse_mean)
    #         error_rate['mape'].append(mape_mean)
    #         error_rate['mae'].append(mae_mean)
    #
    # error_rate = pd.DataFrame(error_rate)
    # toc = time.time()
    # print("Minutes taken = {0:.2f}".format((toc - tic) / 60.0))
    #
    # error_rate
    #
    #
    # # Get optimum value for param and param2, using RMSE
    # temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    # subsample_opt = temp['subsample'].values[0]
    # gamma_opt = temp['gamma'].values[0]
    # print("min RMSE = %0.3f" % error_rate['rmse'].min())
    # print("optimum params = ")
    # subsample_opt, gamma_opt
    #
    #
    # ############ TODO Tuning colsample_bytree & colsample_bylevel
    #
    # param_label = 'colsample_bytree'
    # param_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #
    # param2_label = 'colsample_bylevel'
    # param2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #
    # error_rate = defaultdict(list)
    #
    # tic = time.time()
    # for param in param_list:
    #     for param2 in param2_list:
    #         rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(train_val,
    #                                                               train_size,
    #                                                               N_opt,
    #                                                               H,
    #                                                               seed=model_seed,
    #                                                               n_estimators=n_estimators_opt,
    #                                                               max_depth=max_depth_opt,
    #                                                               learning_rate=learning_rate_opt,
    #                                                               min_child_weight=min_child_weight_opt,
    #                                                               subsample=subsample_opt,
    #                                                               colsample_bytree=param,
    #                                                               colsample_bylevel=param2,
    #                                                               gamma=gamma_opt)
    #
    #         # Collect results
    #         error_rate[param_label].append(param)
    #         error_rate[param2_label].append(param2)
    #         error_rate['rmse'].append(rmse_mean)
    #         error_rate['mape'].append(mape_mean)
    #         error_rate['mae'].append(mae_mean)
    #
    # error_rate = pd.DataFrame(error_rate)
    # toc = time.time()
    # print("Minutes taken = {0:.2f}".format((toc - tic) / 60.0))
    #
    # error_rate
    #
    # # Get optimum value for param and param2, using RMSE
    # temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    # colsample_bytree_opt = temp['colsample_bytree'].values[0]
    # colsample_bylevel_opt = temp['colsample_bylevel'].values[0]
    # print("min RMSE = %0.3f" % error_rate['rmse'].min())
    # print("optimum params = ")
    # colsample_bytree_opt, colsample_bylevel_opt

    ##################### TODO Final Model:

    # Get error metrics on validation set after hyperparameter tuning
    rmse_aft_tuning, mape_aft_tuning, mae_aft_tuning, preds_dict = \
        get_error_metrics(train_val,
                          train_size,
                          N_opt,
                          H,
                          seed=model_seed,
                          n_estimators=n_estimators_opt,
                          max_depth=max_depth_opt,
                          learning_rate=learning_rate_opt)
    print("RMSE = %0.3f" % rmse_aft_tuning)
    print("MAPE = %0.3f%%" % mape_aft_tuning)
    print("MAE = %0.3f" % mae_aft_tuning)

    # Plot validation predictions
    # fig = go.Figure()

    # # Add traces
    # fig.add_trace(go.Scatter(x=train['date'],
    #                          y=train['price'],
    #                          mode='lines',
    #                          name='train',
    #                          line=dict(color='blue')))
    # fig.add_trace(go.Scatter(x=val['date'],
    #                          y=val['price'],
    #                          mode='lines',
    #                          name='validation',
    #                          line=dict(color='orange')))
    # fig.add_trace(go.Scatter(x=test['date'],
    #                          y=test['price'],
    #                          mode='lines',
    #                          name='test',
    #                          line=dict(color='green')))
    #
    # # Plot the predictions
    # n = 0
    # for key in preds_dict:
    #     fig.add_trace(go.Scatter(x=train_val[key:key + H]['date'],
    #                              y=preds_dict[key],
    #                              mode='lines',
    #                              name='predictions',
    #                              line=dict(color=colors[n % len(colors)])))
    #     n = n + 1
    #
    # fig.update_layout(yaxis=dict(title='USD'),
    #                   xaxis=dict(title='date'))
    # fig.update_xaxes(range=['2017-10-16', '2018-11-12'])
    # fig.update_yaxes(range=[127, 157])
    # py.plot(fig, filename='price_val_' + str(sfromdate.date()) + '_' + str(stodate.date()),auto_open=False)

    # Do prediction on test set
    test_rmse_aft_tuning, test_mape_aft_tuning, test_mae_aft_tuning, est, feature_importances, features = \
        get_error_metrics_one_pred(df[pred_day - train_val_size:pred_day + H],
                                   train_size + val_size,
                                   N_opt,
                                   H,
                                   seed=model_seed,
                                   n_estimators=n_estimators_opt,
                                   max_depth=max_depth_opt,
                                   learning_rate=learning_rate_opt)
    print("RMSE = %0.3f" % test_rmse_aft_tuning)
    print("MAPE = %0.3f%%" % test_mape_aft_tuning)
    print("MAE = %0.3f" % test_mae_aft_tuning)

    # Plot test predictions
    fig = go.Figure()

    # # Add traces
    # fig.add_trace(go.Scatter(x=train['date'],
    #                          y=train['price'],
    #                          mode='lines',
    #                          name='train',
    #                          line=dict(color='blue')))
    # fig.add_trace(go.Scatter(x=val['date'],
    #                          y=val['price'],
    #                          mode='lines',
    #                          name='validation',
    #                          line=dict(color='orange')))
    # fig.add_trace(go.Scatter(x=test['date'],
    #                          y=test['price'],
    #                          mode='lines',
    #                          name='test',
    #                          line=dict(color='green')))
    # fig.add_trace(go.Scatter(x=test[:H]['date'],
    #                          y=est,
    #                          mode='lines',
    #                          name='predictions',
    #                          line=dict(color='red')))
    # fig.update_layout(yaxis=dict(title='USD'),
    #                   xaxis=dict(title='date'))
    # py.plot(fig, filename='price_fc_sent_' + str(sfromdate.date()) + '_' + str(stodate.date()), auto_open=False)

    # View a list of the features and their importance scores
    # imp = list(zip(features, feature_importances))
    # imp.sort(key=lambda tup: tup[1], reverse=False)
    # imp

    # # Plot the importance scores as a bar chart
    # fig = go.Figure(go.Bar(
    #     x=[item[1] for item in imp],
    #     y=[item[0] for item in imp],
    #     orientation='h'))
    # fig.update_layout(yaxis=dict(title='feature'),
    #                   xaxis=dict(title='relative importance'))
    # py.plot(fig, filename='imp__sent' + str(sfromdate.date()) + '_' + str(stodate.date()), auto_open=False)

    # Tuned params and before and after error metrics
    d = {'param': ['n_estimators', 'max_depth', 'learning_rate', 'val_rmse', 'val_mape', 'val_mae'],
         'before_tuning': [n_estimators, max_depth, learning_rate, rmse_bef_tuning, mape_bef_tuning, mae_bef_tuning],
         'after_tuning': [n_estimators_opt, max_depth_opt, learning_rate_opt,  rmse_aft_tuning, mape_aft_tuning,
                          mae_aft_tuning]}
    tuned_params = pd.DataFrame(d)
    tuned_params = tuned_params.round(3)
    print(tuned_params)

    toc1 = time.time()
    print("Total minutes taken = {0:.2f}".format((toc1 - tic1) / 60.0))

    print(rmse_bef_tuning, rmse_aft_tuning)

    print(test_rmse_bef_tuning, test_rmse_aft_tuning)

    # # Put results into pickle
    # pickle.dump(rmse_bef_tuning, open(
    #     "./out/nosenti/measures/val_rmse_bef_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + '-' +
    #     df.iloc[pred_day + H]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))
    # pickle.dump(rmse_aft_tuning, open(
    #     "./out/nosenti/measures/val_rmse_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + '-' +
    #     df.iloc[pred_day + H]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))
    # pickle.dump(test_rmse_bef_tuning, open(
    #     "./out/nosenti/measures/test_rmse_bef_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + '-' +
    #     df.iloc[pred_day + H]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))
    # pickle.dump(test_mape_bef_tuning, open(
    #     "./out/nosenti/measures/test_mape_bef_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + '-' +
    #     df.iloc[pred_day + H]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))
    # pickle.dump(test_mae_bef_tuning, open(
    #     "./out/nosenti/measures/test_mae_bef_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + '-' +
    #     df.iloc[pred_day + H]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))
    # pickle.dump(test_rmse_aft_tuning, open(
    #     "./out/nosenti/measures/test_rmse_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + '-' +
    #     df.iloc[pred_day + H]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))
    # pickle.dump(test_mape_aft_tuning, open(
    #     "./out/nosenti/measures/test_mape_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + '-' +
    #     df.iloc[pred_day + H]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))
    # pickle.dump(test_mae_aft_tuning, open(
    #     "./out/nosenti/measures/test_mae_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + '-' +
    #     df.iloc[pred_day + H]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))
    # pickle.dump(est, open(
    #     "./out/nosenti/forecasts/test_est_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + '-' +
    #     df.iloc[pred_day + H]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))

    est = est[-H:] + dflevel[pred_day - 1:pred_day + H - 1]['price'].values
    est = pd.DataFrame(est)
    est.index = test[:H]['date']

    all_fc = pd.concat([all_fc, est])

    pred_day = pred_day + H

all_fc.to_csv(r'Data\XGB_Pred_nosenti.csv')
# pickle.dump(all_fc, open(
#     "./out/nosenti/forecasts/complete-fc.pickle", "wb"))
