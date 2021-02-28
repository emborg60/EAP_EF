# Test for evaluering af hvert forecast og sammenligning mellem forecast
import pandas as pd
import numpy as np
from numpy.random import rand
from numpy import ix_
from itertools import product
import chart_studio.plotly as py
import chart_studio
import plotly.graph_objs as go
import statsmodels.api as sm

chart_studio.tools.set_credentials_file(username='Emborg', api_key='bbBdW78XyA7bPc9shlkf')

np.random.seed(1337)

# Predictions from each forecast
data = pd.read_csv('Data/All_Merged.csv')  # , parse_dates=[0], date_parser=dateparse
data.isna().sum()
data.fillna(0, inplace=True)
data = data.set_index('date')
data = data.loc[~data.index.duplicated(keep='first')]
data = data.drop('2018-10-29')

# Forecasts
LSTM = pd.read_csv('Data/LSTM_Pred.csv', index_col=0)
LSTM = LSTM.loc[~LSTM.index.duplicated(keep='first')]
LSTM = LSTM.iloc[:-11, :]
LSTM = LSTM.drop('2018-10-29')
LSTM_NS = pd.read_csv('Data/LSTM_Pred_NoSent.csv', index_col=0)
LSTM_NS = LSTM_NS.loc[~LSTM_NS.index.duplicated(keep='first')]
LSTM_NS = LSTM_NS.iloc[:-11, :]
LSTM_NS = LSTM_NS.drop('2018-10-29')
ARIMA = pd.read_csv('Data/ARIMA_Pred.csv', index_col=0)
ARIMA = ARIMA.iloc[:-11, :]
ARIMA_NS = pd.read_csv('Data/ARIMA_Pred_NoSent.csv', index_col=0)
ARIMA_NS = ARIMA_NS.iloc[:-11, :]
XGB = pd.read_csv('Data/XGB_Pred.csv', index_col=0)
XGB = XGB.loc[~XGB.index.duplicated(keep='first')]
XGB = XGB.iloc[1:, :]
XGB = XGB.drop('2018-10-29')
XGB_NS = pd.read_csv('Data/XGB_Pred_nosenti.csv', index_col=0)
XGB_NS = XGB_NS.loc[~XGB_NS.index.duplicated(keep='first')]
XGB_NS = XGB_NS.iloc[1:, :]
XGB_NS = XGB_NS.drop('2018-10-29')
AR1 = pd.read_csv('Data/AR1.csv', index_col=0)
AR1 = AR1.iloc[:-11, :]
VAR = pd.read_csv('Data/VAR_pred.csv', index_col=0)
VAR = VAR.loc[~VAR.index.duplicated(keep='first')]
VAR = VAR[VAR.index.isin(LSTM.index)]['price']
VAR_NS = pd.read_csv('Data/VAR_pred_nosenti.csv', index_col=0)
VAR_NS = VAR_NS.loc[~VAR_NS.index.duplicated(keep='first')]
VAR_NS = VAR_NS[VAR_NS.index.isin(LSTM.index)]['price']

# Price for the forecasting period
price = data[data.index.isin(LSTM.index)]
price = price[['price']]
ARIMA.index = price.index
ARIMA_NS.index = price.index
XGB.index = price.index
XGB_NS.index = price.index

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

# Combined Forecast DataFrame
fc = pd.DataFrame()
fc = price
fc = fc.merge(AR1[['forecast']], how='left', left_index=True, right_index=True)
fc = fc.merge(ARIMA[['forecast']], how='left', left_index=True, right_index=True)
fc = fc.merge(ARIMA_NS[['forecast']], how='left', left_index=True, right_index=True)
fc = fc.merge(VAR, how='left', left_index=True, right_index=True)
fc = fc.merge(VAR_NS, how='left', left_index=True, right_index=True)
fc = fc.merge(XGB, how='left', left_index=True, right_index=True)
fc = fc.merge(XGB_NS, how='left', left_index=True, right_index=True)
fc = fc.merge(LSTM[['LSTM']], how='left', left_index=True, right_index=True)
fc = fc.merge(LSTM_NS[['LSTM']], how='left', left_index=True, right_index=True)

# fc = fc.merge(XGB_NS, how='left', left_index=True, right_index=True)
fc.columns = ['Price', 'AR1', 'ARIMAX', 'ARIMAX_NS', 'VAR', 'VAR_NS', 'XGB', 'XGB_NS', 'LSTM', 'LSTM_NS']
# fc.to_csv(r'Data\All_Forecasts.csv')

fig = go.Figure()
n = 0
for key in fc.columns:
    fig.add_trace(go.Scatter(x=fc.index,
                             y=fc[key],
                             mode='lines',
                             name=key,
                             line=dict(color=colors[n % len(colors)])))
    n = n + 1
fig.update_layout(yaxis=dict(title='USD'),
                  xaxis=dict(title='date'))
py.plot(fig, filename='price_all_fc')

# Actual price
actual = fc[['Price']]
fc = fc.iloc[:, 1:]


# Error metrics
def RMSE(fc, actual):
    actual = actual.values
    fc = fc.values
    losses = fc - actual
    RMSE = np.sqrt(np.mean(losses ** 2, axis=0))
    return (RMSE)


def MAE(fc, actual):
    actual = actual.values
    fc = fc.values
    losses = fc - actual
    MAE = np.mean(np.abs(losses), axis=0)
    return (MAE)



def residual_bar_plot(fc_1, fc_2, actuals, name1, name2):

    df = pd.DataFrame(fc_1.values - actuals.values)
    df[name2] = fc_2.values - actuals.values
    df.columns = [name1,name2]
    df.hist()
    print(name1)
    print(round(sm.tsa.stattools.adfuller(df[name1])[1],4))
    print(round(sm.stats.stattools.jarque_bera(df[name1])[1],4))
    print(name2)
    print(round(sm.tsa.stattools.adfuller(df[name2])[1],4))
    print(round(sm.stats.stattools.jarque_bera(df[name2])[1],4))

residual_bar_plot(fc[['ARIMAX']], fc[['ARIMAX_NS']], actual, 'ARIMA', 'ARIMA_NS')
residual_bar_plot(fc[['LSTM']], fc[['LSTM_NS']], actual, 'LSTM', 'LSTM_NS')
residual_bar_plot(fc[['VAR']], fc[['VAR_NS']], actual, 'VAR', 'VAR_NS')
residual_bar_plot(fc[['XGB']], fc[['XGB_NS']], actual, 'XGB', 'XGB_NS')



name1 = 'ARIMAX'
fc_1 = fc[['ARIMAX']]

# split_date = '2019-05-01'
# fc = fc.loc[fc.index >= split_date]
# actual = actual.loc[actual.index >= split_date]

rmse = RMSE(fc, actual)
mae = MAE(fc, actual)


print(pd.DataFrame(rmse).to_latex())

# Diebold Mariano testing
dm_result = list()
done_models = list()
models_list = fc.columns
for model1 in models_list:
    for model2 in models_list:
        if model1 != model2:
            dm_result.append(dm_test(fc[[model1]], fc[[model2]], actual))

dm_result = pd.DataFrame(dm_result)
# dm_result['t-stat'] = np.abs(dm_result['t-stat'])
dm_result = dm_result.loc[~np.abs(dm_result['t-stat']).duplicated(keep='first')]
dm_result['t-stat'] = round(dm_result['t-stat'],2)
dm_result['p-value'] = round(dm_result['p-value'],4)
print(dm_result.to_latex())

# Clark West
cw1 = cw_test(ARIMA, ARIMA_NS, actual)
print(cw1)
cw2 = cw_test(LSTM[['LSTM']], LSTM_NS[['LSTM']], actual)
print(cw2)
cw3 = cw_test(XGB[['est']], XGB_NS[['est']], actual)
print(cw3)


cspe_plot(fc[['XGB_NS']], fc[['XGB']], actual)

# Model Confidence Set
# https://michael-gong.com/blogs/model-confidence-set/?fbclid=IwAR38oo302TSJ4BFqTpluh5aeivkyM6A1cc0tnZ_JUX08PNwRzQkIi4WPlps

# Wrap data and compute the Mean Absolute Error
MCS_data = pd.DataFrame(np.c_[fc.AR1, fc.ARIMAX, fc.ARIMAX_NS, fc.LSTM, fc.LSTM_NS, fc.VAR, fc.VAR_NS, fc.XGB, fc.XGB_NS, actual.Price],
                        columns=['AR1','ARIMAX', 'ARIMAX_NS', 'LSTM', 'LSTM_NS','VAR','VAR_NS','XGB','XGB_NS', 'Actual'])
losses = pd.DataFrame()
for model in MCS_data.columns: #['ARIMA', 'ARIMA_NS', 'LSTM', 'LSTM_NS']:
    losses[model] = np.abs(MCS_data[model] - MCS_data['Actual'])
losses=losses.iloc[:,:-1]
mcs = ModelConfidenceSet(losses, 0.1, 3, 1000).run()
mcs.included
mcs.pvalues

# Forecast combinations
fc.columns[1:]


l1 = fc.columns[1:].values
l2 = ['ARIMAX', 'VAR', 'XGB','LSTM']
l3 = ['ARIMAX_NS', 'VAR_NS', 'XGB_NS','LSTM_NS']

comb_results = pd.DataFrame([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
comb_results.index = ['All','S','NS']
comb_results.columns = ['Equal', 'MSE', 'Rank', 'Time(1)','Time(7)']
l_list = [l1,l2,l3]
i = 0
for l in l_list:
    print(l)
    pred = fc[l]

    # Combinations
    eq = fc_comb(actual=actual, fc=pred, weights="equal")
    #bgw = fc_comb(actual=actual, fc=fc[fc.columns[1:]], weights="BGW")
    mse = fc_comb(actual=actual, fc=pred, weights="MSE")
    rank = fc_comb(actual=actual, fc=pred, weights="rank")
    time = fc_comb(actual=actual, fc=pred, weights="time")
    time7 = fc_comb(actual=actual, fc=pred, weights="time", window=7)
    time14 = fc_comb(actual=actual, fc=pred, weights="time", window=14)
    time30 = fc_comb(actual=actual, fc=pred, weights="time", window=30)
    time60 = fc_comb(actual=actual, fc=pred, weights="time", window=60)



    comb_results.iloc[i,0] = MAE(eq, actual)
    comb_results.iloc[i,1] = MAE(mse, actual)
    comb_results.iloc[i,2] = MAE(rank, actual)
    comb_results.iloc[i,3] = MAE(time, actual)
    comb_results.iloc[i,4] = MAE(time7, actual)
    i = i + 1


print(round(comb_results,2).to_latex())

rank = pd.DataFrame(rank)
rank.columns = ['Rank']
eq = pd.DataFrame(eq)
eq.columns = ['Eq']



dm_test(rank[['Rank']], eq[['Eq']], actual)




# Fun
# ctions

# Diebold Mariano test function
def dm_test(fc, fc_nested, actual):
    fc_name = fc.columns[0]
    fc_nested_name = fc_nested.columns[0]

    import statsmodels.formula.api as smf
    from sklearn.metrics import mean_squared_error
    fc = fc.values
    fc_nested = fc_nested.values
    actual = price.values
    e_fc = actual - fc
    e_nested = actual - fc_nested
    f_dm = e_nested ** 2 - e_fc ** 2
    f_dm = pd.DataFrame(f_dm, columns=['f_dm'])
    nwResult = smf.ols('f_dm ~ 1', data=f_dm).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    dm_out = dict()
    dm_out['t-stat'] = nwResult.tvalues[0]
    dm_out['p-value'] = round(nwResult.pvalues[0], 4)

    if dm_out['p-value'] < 0.05:
        if mean_squared_error(actual, fc) < mean_squared_error(actual, fc_nested):
            dm_out['conclusion'] = 'First forecast is best ' + fc_name + ' better then ' + fc_nested_name
        else:
            dm_out['conclusion'] = 'Second forecast is best' + fc_nested_name + ' better then ' + fc_name
    else:
        dm_out['conclusion'] = 'Forecasts have equal predictive power between ' + fc_nested_name + ' and ' + fc_name
    return dm_out


# Clark West test function
def cw_test(fc, fc_nested, actual):
    import statsmodels.formula.api as smf
    from sklearn.metrics import mean_squared_error
    fc = fc.values
    fc_nested = fc_nested.values
    actual = price.values
    e_fc = actual - fc
    e_nested = actual - fc_nested
    e_diff = fc - fc_nested
    f_CW = e_nested ** 2 - e_fc ** 2 + e_diff ** 2
    f_CW = pd.DataFrame(f_CW, columns=['f_CW'])

    nwResult = smf.ols('f_CW ~ 1', data=f_CW).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    cw_out = dict()
    cw_out['t-stat'] = nwResult.tvalues[0]
    cw_out['p-value'] = nwResult.pvalues[0]

    if cw_out['p-value'] < 0.05:
        if mean_squared_error(actual, fc) < mean_squared_error(actual, fc_nested):
            cw_out['conclusion'] = 'Non-nested is best'
        else:
            cw_out['conclusion'] = 'Nested is best'
    else:
        cw_out['conclusion'] = 'Forecasts have equal predictive power '
    return cw_out


# Cumulative Squared Predictive Error
def cspe_plot(benchmark, shot, actual):
    import chart_studio.plotly as py
    import chart_studio
    import plotly.graph_objs as go
    chart_studio.tools.set_credentials_file(username='Emborg', api_key='bbBdW78XyA7bPc9shlkf')

    cumshot = np.cumsum((actual.values - shot.values) ** 2) - np.cumsum((actual.values - benchmark.values) ** 2)
    fig = go.Figure()
    fig.add_shape(
        # Line Horizontal
        type="line",
        x0=price.index[0],
        x1=price.index[-1],
        line=dict(
            color="LightSeaGreen",
            width=1,
        ),
    )
    fig.update_layout(yaxis=dict(title='Cummulative squared prediction error'),
                      xaxis=dict(title='date'))
    fig.add_trace(go.Scatter(x=price.index,
                             y=cumshot,
                             mode='lines',
                             name='CSPE',
                             line=dict(color='blue')))

    py.plot(fig, filename='cumsum')


def fc_error(fc, actual, name):
    import chart_studio.plotly as py
    import chart_studio
    import plotly.graph_objs as go
    chart_studio.tools.set_credentials_file(username='Emborg', api_key='bbBdW78XyA7bPc9shlkf')

    fig = go.Figure()
    fig.add_shape(
        # Line Horizontal
        type="line",
        x0=actual.index[0],
        x1=actual.index[-1],
        line=dict(
            color="LightSeaGreen",
            width=1,
        ),
    )
    fig.update_layout(yaxis=dict(title='Forecast error'),
                      xaxis=dict(title='date'))
    fig.add_trace(go.Scatter(x=actual.index,
                             y=error,
                             mode='lines',
                             name='Residuals',
                             line=dict(color='blue')))
    py.plot(fig, filename=name)


# Forecast combination
def fc_comb(actual, fc, weights="equal", window=1):
    # Input:
    # actual: Tx1 dataframe of actual values
    # fc: TxN dataframe of N predictions
    # Weights: Weighting scheme, either equal (equal), Bates & Granger Weighted (BGW), MSE weights (MSE)
    #          , ranked (rank), and Time-varying weighting scheme (time)
    # weight_vector = pd.Dataframe()
    actual = actual.values
    fc = fc.values
    losses = fc - actual
    MSE = np.mean(losses ** 2, axis=0)
    Ones = np.ones((1, fc.shape[1]), dtype=int).T
    if (weights == "equal"):
        weight_vector = Ones / fc.shape[1]
    elif (weights == "BGW"):
        print("Not Available")
        # Out = pd.DataFrame()
        # weight_vector = Ones / fc.shape[1]
        # est = pd.DataFrame(weight_vector.T @ fc[0:3, :].T).T
        # Out = pd.concat([Out, est], ignore_index=True)
        # for i in range(2, actual.shape[0] - 1):
        #     cov = np.cov(losses[0:i + 1, :].T)
        #     OnesBGW = np.ones((1, cov.shape[1]), dtype=int).T
        #     weight_vector = np.linalg.inv(OnesBGW.T @ np.linalg.inv(cov) @ OnesBGW) * np.linalg.inv(cov) @ OnesBGW
        #     est = pd.Series(weight_vector.T @ fc[i + 1, :])
        #     Out = pd.concat([Out, est], ignore_index=True)
    elif (weights == "MSE"):
        Out = pd.DataFrame()
        weight_vector = Ones / fc.shape[1]
        est = pd.Series(weight_vector.T @ fc[0, :])
        Out = pd.concat([Out, est], ignore_index=True)
        for i in range(0, actual.shape[0]-1):
            weight_vector = 1 / sum(losses[0:i + 1, :] ** 2) / (sum(1 / sum(losses[0:i + 1, :] ** 2)))
            est = pd.Series(weight_vector @ fc[i+1, :])
            Out = pd.concat([Out, est], ignore_index=True)
    elif (weights == "rank"):
        from scipy.stats import rankdata
        Out = pd.DataFrame()
        weight_vector = Ones / fc.shape[1]
        est = pd.Series(weight_vector.T @ fc[0, :])
        Out = pd.concat([Out, est], ignore_index=True)
        for i in range(0, actual.shape[0] - 1):
            MSE = np.mean(losses[0:i + 1, :] ** 2, axis=0)
            rank = rankdata(MSE, method='min')
            weight_vector = 1 / rank / (sum(1 / rank))
            est = pd.Series(weight_vector @ fc[i + 1, :])
            Out = pd.concat([Out, est], ignore_index=True)
    elif (weights == "time"):
        Out = pd.DataFrame()
        weight_vector = Ones / fc.shape[1]
        est = pd.Series(weight_vector.T @ fc[0, :])
        Out = pd.concat([Out, est], ignore_index=True)
        j = 1
        for i in range(0, window - 1):
            weight_vector = 1 / sum(losses[i - j + 1:i + 1, :] ** 2) / (sum(1 / sum(losses[i - j + 1:i + 1, :] ** 2)))
            est = pd.Series(weight_vector @ fc[i+1, :])
            Out = pd.concat([Out, est], ignore_index=True)
            j = j + 1

        v = window
        i = window - 1
        while i < len(losses)-1:
            weight_vector = 1 / sum(losses[i - v + 1:i + 1, :] ** 2) / (sum(1 / sum(losses[i - v + 1:i + 1, :] ** 2)))
            est = pd.Series(weight_vector @ fc[i+1, :])
            Out = pd.concat([Out, est], ignore_index=True)
            i = i + 1

    if (weights == "equal"):
        Output = pd.DataFrame((weight_vector.T @ fc.T).T)
    else:
        Output = Out

    return (Output)


# Model Confidence Set
def bootstrap_sample(data, B, w):
    '''
    Bootstrap the input data
    data: input numpy data array
    B: boostrap size
    w: block length of the boostrap
    '''
    t = len(data)
    p = 1 / w
    indices = np.zeros((t, B), dtype=int)
    indices[0, :] = np.ceil(t * rand(1, B))
    select = np.asfortranarray(rand(B, t).T < p)
    vals = np.ceil(rand(1, np.sum(np.sum(select))) * t).astype(int)
    indices_flat = indices.ravel(order="F")
    indices_flat[select.ravel(order="F")] = vals.ravel()
    indices = indices_flat.reshape([B, t]).T
    for i in range(1, t):
        indices[i, ~select[i, :]] = indices[i - 1, ~select[i, :]] + 1
    indices[indices > t] = indices[indices > t] - t
    indices -= 1
    return data[indices]


def compute_dij(losses, bsdata):
    '''Compute the loss difference'''
    t, M0 = losses.shape
    B = bsdata.shape[1]
    dijbar = np.zeros((M0, M0))
    for j in range(M0):
        dijbar[j, :] = np.mean(losses - losses[:, [j]], axis=0)

    dijbarstar = np.zeros((B, M0, M0))
    for b in range(B):
        meanworkdata = np.mean(losses[bsdata[:, b], :], axis=0)
        for j in range(M0):
            dijbarstar[b, j, :] = meanworkdata - meanworkdata[j]

    vardijbar = np.mean((dijbarstar - np.expand_dims(dijbar, 0)) ** 2, axis=0)
    vardijbar += np.eye(M0)

    return dijbar, dijbarstar, vardijbar


def calculate_PvalR(z, included, zdata0):
    '''Calculate the p-value of relative algorithm'''
    empdistTR = np.max(np.max(np.abs(z), 2), 1)
    zdata = zdata0[ix_(included - 1, included - 1)]
    TR = np.max(zdata)
    pval = np.mean(empdistTR > TR)
    return pval


def calculate_PvalSQ(z, included, zdata0):
    '''Calculate the p-value of sequential algorithm'''
    empdistTSQ = np.sum(z ** 2, axis=1).sum(axis=1) / 2
    zdata = zdata0[ix_(included - 1, included - 1)]
    TSQ = np.sum(zdata ** 2) / 2
    pval = np.mean(empdistTSQ > TSQ)
    return pval


def iterate(dijbar, dijbarstar, vardijbar, alpha, algorithm="R"):
    '''Iteratively excluding inferior model'''
    B, M0, _ = dijbarstar.shape
    z0 = (dijbarstar - np.expand_dims(dijbar, 0)) / np.sqrt(
        np.expand_dims(vardijbar, 0)
    )
    zdata0 = dijbar / np.sqrt(vardijbar)

    excludedR = np.zeros([M0, 1], dtype=int)
    pvalsR = np.ones([M0, 1])

    for i in range(M0 - 1):
        included = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
        m = len(included)
        z = z0[ix_(range(B), included - 1, included - 1)]

        if algorithm == "R":
            pvalsR[i] = calculate_PvalR(z, included, zdata0)
        elif algorithm == "SQ":
            pvalsR[i] = calculate_PvalSQ(z, included, zdata0)

        scale = m / (m - 1)
        dibar = np.mean(dijbar[ix_(included - 1, included - 1)], 0) * scale
        dibstar = np.mean(dijbarstar[ix_(range(B), included - 1, included - 1)], 1) * (
                m / (m - 1)
        )
        vardi = np.mean((dibstar - dibar) ** 2, axis=0)
        t = dibar / np.sqrt(vardi)
        modeltoremove = np.argmax(t)
        excludedR[i] = included[modeltoremove]

    maxpval = pvalsR[0]
    for i in range(1, M0):
        if pvalsR[i] < maxpval:
            pvalsR[i] = maxpval
        else:
            maxpval = pvalsR[i]

    excludedR[-1] = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
    pl = np.argmax(pvalsR > alpha)
    includedR = excludedR[pl:]
    excludedR = excludedR[:pl]
    return includedR - 1, excludedR - 1, pvalsR


def MCS(losses, alpha, B, w, algorithm):
    '''Main function of the MCS'''
    t, M0 = losses.shape
    bsdata = bootstrap_sample(np.arange(t), B, w)
    dijbar, dijbarstar, vardijbar = compute_dij(losses, bsdata)
    includedR, excludedR, pvalsR = iterate(
        dijbar, dijbarstar, vardijbar, alpha, algorithm=algorithm
    )
    return includedR, excludedR, pvalsR


class ModelConfidenceSet(object):
    def __init__(self, data, alpha, B, w, algorithm="SQ", names=None):
        """
        Implementation of Econometrica Paper:
        Hansen, Peter R., Asger Lunde, and James M. Nason. "The model confidence set." Econometrica 79.2 (2011): 453-497.

        Input:
            data->pandas.DataFrame or numpy.ndarray: input data, columns are the losses of each model
            alpha->float: confidence level
            B->int: bootstrap size for computation covariance
            w->int: block size for bootstrap sampling
            algorithm->str: SQ or R, SQ is the first t-statistics in Hansen (2011) p.465, and R is the second t-statistics
            names->list: the name of each model (corresponding to each columns).

        Method:
            run(self): compute the MCS procedure

        Attributes:
            included: models that are in the model confidence sets at confidence level of alpha
            excluded: models that are NOT in the model confidence sets at confidence level of alpha
            pvalues: the bootstrap p-values of each models
        """

        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.names = data.columns.values if names is None else names
        elif isinstance(data, np.ndarray):
            self.data = data
            self.names = np.arange(data.shape[1]) if names is None else names

        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must be larger than zero and less than 1, found {alpha}"
            )
        if not isinstance(B, int):
            try:
                B = int(B)
            except Exception as identifier:
                raise RuntimeError(
                    f"Bootstrap size B must be a integer, fail to convert", identifier
                )
        if B < 1:
            raise ValueError(f"Bootstrap size B must be larger than 1, found {B}")
        if not isinstance(w, int):
            try:
                w = int(w)
            except Exception as identifier:
                raise RuntimeError(
                    f"Bootstrap block size w must be a integer, fail to convert",
                    identifier,
                )
        if w < 1:
            raise ValueError(f"Bootstrap block size w must be larger than 1, found {w}")

        if algorithm not in ["R", "SQ"]:
            raise TypeError(f"Only R and SQ algorithm supported, found {algorithm}")

        self.alpha = alpha
        self.B = B
        self.w = w
        self.algorithm = algorithm

    def run(self):
        included, excluded, pvals = MCS(
            self.data, self.alpha, self.B, self.w, self.algorithm
        )

        self.included = self.names[included].ravel().tolist()
        self.excluded = self.names[excluded].ravel().tolist()
        self.pvalues = pd.Series(pvals.ravel(), index=self.excluded + self.included)
        return self
