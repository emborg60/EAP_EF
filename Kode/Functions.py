import re
import pandas as pd
from google.oauth2 import service_account
from langdetect import detect_langs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
import numpy as np
from numpy import mat, mean, sqrt, diag
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


def language_filter(df, series=str, language_select=str):
    df = df.reset_index(drop=True)

    # Purpose: Detect language of string using Google's langDetect
    # Arguments: DF; DataFrame, (series = name of column in df as string), (language_select = two letter
    # string of language code that you want)

    df_copy = df.copy()

    df_copy['language'] = df_copy[series].apply(detect_langs)

    # new column ['contains_your_language'] returns 'True' if ['language'] contains any probability of your language
    df_copy['contains_your_language'] = df_copy['language'].apply(str).str.contains(language_select)

    # parse data to only return values where ['contains_your_language'] is True
    df_copy = df_copy.loc[df_copy['contains_your_language'] == True]

    # remove ['language'] and ['contains_your_language'] as they are no longer needed
    del df_copy['language']
    del df_copy['contains_your_language']

    # reindex df
    df_copy = df_copy.reset_index(drop=True)

    # return your new filtered DataFrame
    return df_copy


def get_sentiment(df, series=str):
    # initialize sentiment classifier
    sia = SIA()

    # get sentiment
    sentiment = df[series].apply(sia.polarity_scores)

    # create sentiment df
    sentiment = pd.DataFrame(sentiment.tolist())

    # merge sentiment with your df

    df = df.merge(sentiment, how='left', left_index=True, right_index=True)
    df['sentiment'] = df['compound'].apply(categorize_sentiment)
    df['sentiment'] = pd.Categorical(df['sentiment'])
    binary_sentiment = df['sentiment'].str.get_dummies()

    df = df.merge(binary_sentiment, how='left', left_index=True, right_index=True)
    return df


def categorize_sentiment(x):
    if x >= 0.05:
        return 'positive_comment'
    elif 0.05 > x > -0.05:
        return 'neutral_comment'
    elif -0.05 >= x:
        return 'negative_comment'


def group_sentiment(dfSentiment):
    dfSentiment['datetime'] = pd.to_datetime(dfSentiment['created_utc'], unit='s')
    dfSentiment['date'] = pd.DatetimeIndex(dfSentiment['datetime']).date

    dfSentiment = dfSentiment[
        ['created_utc', 'negative_comment', 'neutral_comment', 'positive_comment', 'datetime', 'date']]

    dfSentiment = dfSentiment.groupby(by=['date']).sum()

    return dfSentiment


def collect_big_query(sQuery):
    credentials = service_account.Credentials.from_service_account_file(r'insert-file-path-for-json-creditial-file')
    project_id = 'insert-project-id-here'

    data = pd.read_gbq(sQuery, project_id=project_id, credentials=credentials, dialect='standard')

    return data


# Funktion til Fama-MacBeth 2-pass regression der skal bruges i Fama-French Framework

def FMB(returns, riskFactors):
    # function  fmbOut = famaMacBeth(returns,riskFactors)
    # Purpose:  Estimate linear asset pricing models using the Fama and MacBeth
    #           (1973) two-pass cross-sectional regression methodology.
    #
    # Input:    returns     = TxN maxtrix of portfolio excess returns
    #           riskFactors = TxK matrix of common risk factors
    #
    # Output:   A struct including results from the two steps

    # Use mat for easier linear algebra
    factors = mat(riskFactors.values)
    excessReturns = mat(returns.values)  # Måske ikke .values

    # Shape information
    t, n = excessReturns.shape

    # Time series regressions
    X = sm.add_constant(factors)  # Laver X ved at inkludere en 1 vektor på faktorer
    ts_res = sm.OLS(excessReturns, X).fit()  # First pass regression
    beta = ts_res.params[1:]

    # Cross-section regression
    cs_params = pd.DataFrame()
    cs_X = sm.add_constant(beta.T)
    for iObs in range(t):
        cs_params = pd.concat([cs_params, pd.DataFrame(sm.OLS(excessReturns[iObs].T, cs_X).fit().params)], axis=1)

    # Risk prices and Fama-MacBeth standard errors and t-stats
    RiskPrices = cs_params.mean(axis=1).T
    seGamma = sqrt((cs_params.T.sub(RiskPrices) ** 2).sum(axis=0) / t ** 2)
    tGamma = RiskPrices / seGamma

    # Mean and fitted excess returns
    meanReturns = pd.DataFrame(mean(excessReturns, 0))
    fittedValues = (pd.DataFrame(cs_X) @ pd.DataFrame(RiskPrices)).T

    # Cross sectional R^2
    Ones = pd.DataFrame(np.ones((1, n), dtype=int)).T
    errResid = meanReturns - fittedValues
    s2 = mean(errResid ** 2, axis=1)
    vary = mean((meanReturns.T - Ones * mean(meanReturns, axis=1)) ** 2)
    rSquared = 100 * (1 - s2 / vary)

    fmbOut = dict()
    fmbOut['beta'] = ts_res.params
    fmbOut['gamma'] = RiskPrices
    fmbOut['se'] = seGamma
    fmbOut['tstat'] = tGamma
    fmbOut['r2'] = rSquared
    fmbOut['fit'] = fittedValues
    fmbOut['mean'] = meanReturns
    return fmbOut


def PortfolioSort(dfReturns, dfMarketCap, dfSignal):
    dfSignalSorted = pd.DataFrame()

    dfReturns = dfReturns[1:]
    dfMarketCap = dfMarketCap[1:]
    dfSignal = dfSignal[:-1]

    # Lag returns based on portfolio decision
    dfReturns.index = dfSignal.index
    dfMarketCap.index = dfSignal.index

    # Calculate Number of current coins in market portfolio:
    MarketCapDummy = dfMarketCap.iloc[:, :].ge(0.1, axis=0)
    MarketCapDummy = MarketCapDummy.where(MarketCapDummy == 1, np.nan)

    dfSignal = dfSignal.multiply(MarketCapDummy)
    NumActiveCoins = dfSignal.iloc[:, :].ge(-1.1, axis=0).sum(axis=1)

    # Rank top based on signal
    df_rank = dfSignal.stack(dropna=False).groupby(level=0).rank(ascending=False, method='first').unstack()
    dfSignal_Trank = df_rank.le(round(NumActiveCoins / 3), axis=0)

    # Get top Market cap and returns
    dfMarketCap_Top = dfMarketCap[dfSignal_Trank]
    dfReturns_Top = dfReturns[dfSignal_Trank]
    dfMarketCap_Top = dfMarketCap_Top.fillna(0)
    dfReturns_Top = dfReturns_Top.fillna(0)

    # Get bottun based on signal
    df_rank = dfSignal.stack(dropna=False).groupby(level=0).rank(ascending=True, method='first').unstack()
    dfSignal_Brank = df_rank.le(round(NumActiveCoins / 3), axis=0)

    # get bottom market cap and returns
    dfMarketCap_Low = dfMarketCap[dfSignal_Brank]
    dfReturns_Low = dfReturns[dfSignal_Brank]
    dfMarketCap_Low = dfMarketCap_Low.fillna(0)
    dfReturns_Low = dfReturns_Low.fillna(0)

    dfReturns_Mid = dfReturns.sub(dfReturns_Top)
    dfReturns_Mid = dfReturns_Mid.sub(dfReturns_Low)
    dfMarketCap_Mid = dfMarketCap.sub(dfMarketCap_Top)
    dfMarketCap_Mid = dfMarketCap_Mid.sub(dfMarketCap_Low)

    dfReturns_Mid = dfReturns_Mid.fillna(0)
    dfMarketCap_Mid = dfMarketCap_Mid.fillna(0)

    dfSignalSorted['Low'] = dfReturns_Low.multiply(dfMarketCap_Low).sum(axis=1) / dfMarketCap_Low.sum(axis=1)
    dfSignalSorted['Mid'] = dfReturns_Mid.multiply(dfMarketCap_Mid).sum(axis=1) / dfMarketCap_Mid.sum(axis=1)
    dfSignalSorted['Top'] = dfReturns_Top.multiply(dfMarketCap_Top).sum(axis=1) / dfMarketCap_Top.sum(axis=1)
    dfSignalSorted['LS'] = dfSignalSorted['Top'] - dfSignalSorted['Low']

    return dfSignalSorted


def FactorSort(dfReturns, dfMarketCap, dfSignal):
    dfSignalSorted = pd.DataFrame()

    dfReturns = dfReturns[1:]
    dfMarketCap = dfMarketCap[1:]
    dfSignal = dfSignal[:-1]

    # Calculate Number of current coins in market portfolio:
    MarketCapDummy = dfMarketCap.iloc[:, :].ge(0.1, axis=0)
    MarketCapDummy = MarketCapDummy.where(MarketCapDummy == 1, np.nan)

    dfSignal = dfSignal.multiply(MarketCapDummy)
    NumActiveCoins = dfSignal.iloc[:, :].ge(-1.1, axis=0).sum(axis=1)

    # Rank top based on signal
    df_rank = dfSignal.stack(dropna=False).groupby(level=0).rank(ascending=False, method='first').unstack()
    dfSignal_Trank = df_rank.le(round(NumActiveCoins * 0.3), axis=0)

    # Get top Market cap and returns

    dfMarketCap_Top = dfMarketCap[dfSignal_Trank]
    dfReturns_Top = dfReturns[dfSignal_Trank]
    dfMarketCap_Top = dfMarketCap_Top.fillna(0)
    dfReturns_Top = dfReturns_Top.fillna(0)

    # Get bottun based on signal
    df_rank = dfSignal.stack(dropna=False).groupby(level=0).rank(ascending=True, method='first').unstack()
    dfSignal_Brank = df_rank.le(round(NumActiveCoins * 0.3), axis=0)

    # get bottom market cap and returns
    dfMarketCap_Low = dfMarketCap[dfSignal_Brank]
    dfReturns_Low = dfReturns[dfSignal_Brank]
    dfMarketCap_Low = dfMarketCap_Low.fillna(0)
    dfReturns_Low = dfReturns_Low.fillna(0)

    dfReturns_Mid = dfReturns.sub(dfReturns_Top)
    dfReturns_Mid = dfReturns_Mid.sub(dfReturns_Low)
    dfMarketCap_Mid = dfMarketCap.sub(dfMarketCap_Top)
    dfMarketCap_Mid = dfMarketCap_Mid.sub(dfMarketCap_Low)

    dfReturns_Mid = dfReturns_Mid.fillna(0)
    dfMarketCap_Mid = dfMarketCap_Mid.fillna(0)

    dfSignalSorted['Low'] = dfReturns_Low.multiply(dfMarketCap_Low).sum(axis=1) / dfMarketCap_Low.sum(axis=1)
    dfSignalSorted['Mid'] = dfReturns_Mid.multiply(dfMarketCap_Mid).sum(axis=1) / dfMarketCap_Mid.sum(axis=1)
    dfSignalSorted['Top'] = dfReturns_Top.multiply(dfMarketCap_Top).sum(axis=1) / dfMarketCap_Top.sum(axis=1)
    dfSignalSorted['LS'] = dfSignalSorted['Top'] - dfSignalSorted['Low']

    return dfSignalSorted


def ReturnSignificance(dfReturns):
    # Returns: Tx5 matrix of Low, Mid1, Mid2, Top and LS returns of portfolio strategy

    Ones = pd.DataFrame(np.ones((1, dfReturns.shape[0]), dtype=int)).T
    Ones.index = dfReturns.index
    Low_res = sm.OLS(dfReturns['P1'], Ones).fit()
    Mid1_res = sm.OLS(dfReturns['P2'], Ones).fit()
    Mid2_res = sm.OLS(dfReturns['P3'], Ones).fit()
    Top_res = sm.OLS(dfReturns['P4'], Ones).fit()
    LS_res = sm.OLS(dfReturns['LS'], Ones).fit()
    Values = [[Low_res.params, Mid1_res.params, Mid2_res.params, Top_res.params, LS_res.params]]
    Values.append([Low_res.bse, Mid1_res.bse, Mid2_res.bse, Top_res.bse, LS_res.bse])
    Values.append([Low_res.tvalues, Mid1_res.tvalues, Mid2_res.tvalues, Top_res.tvalues, LS_res.tvalues])
    Values.append([Low_res.pvalues, Mid1_res.pvalues, Mid2_res.pvalues, Top_res.pvalues, LS_res.pvalues])
    df = pd.DataFrame(Values, columns=['P1', 'P2', 'P3', 'P4', 'P4'], index=['beta', 'se', 't-values', 'p-values'], dtype=np.float64)
    print(LS_res.summary())
    return df

def ReturnSignificance2(dfReturns):
    # Returns: Tx5 matrix of Low, Mid1, Mid2, Top and LS returns of portfolio strategy

    Ones = pd.DataFrame(np.ones((1, dfReturns.shape[0]), dtype=int)).T
    Ones.index = dfReturns.index
    Low_res = sm.OLS(dfReturns['P1'], Ones).fit()
    Mid_res = sm.OLS(dfReturns['P2'], Ones).fit()
    Top_res = sm.OLS(dfReturns['P3'], Ones).fit()
    LS_res = sm.OLS(dfReturns['LS'], Ones).fit()
    Values = [[Low_res.params, Mid_res.params, Top_res.params, LS_res.params]]
    Values.append([Low_res.bse, Mid_res.bse, Top_res.bse, LS_res.bse])
    Values.append([Low_res.tvalues, Mid_res.tvalues, Top_res.tvalues, LS_res.tvalues])
    Values.append([Low_res.pvalues, Mid_res.pvalues, Top_res.pvalues, LS_res.pvalues])
    df = pd.DataFrame(Values, columns=['P1', 'P2', 'P3', 'LS'], index=['beta', 'se', 't-values', 'p-values'], dtype=np.float64)
    print(LS_res.summary())
    return df

def FMB_Shank(returns, riskFactors, nLagsTS):
    # function  fmbOut = famaMacBeth(returns,riskFactors)
    # Purpose:  Estimate linear asset pricing models using the Fama and MacBeth
    #           (1973) two-pass cross-sectional regression methodology.
    #
    # Input:    returns     = TxN maxtrix of portfolio excess returns
    #           riskFactors = TxK matrix of common risk factors
    #           nLagsTS     = Scalar indicating the number of lags to include in HAC
    #                         estimator of variance in first-stage regression
    #
    # Output:  Two structures including results from the two steps

    # Use mat for easier linear algebra
    factors = mat(riskFactors.values)
    excessReturns = mat(returns.values)  # Måske ikke .values

    # Shape information
    t, n = excessReturns.shape
    nFactors = factors.shape[1]

    # Time series regressions
    # X = sm.add_constant(factors) # Laver X ved at inkludere en 1 vektor på faktorer
    # ts_res = sm.OLS(excessReturns, X).fit() # First pass regression # Gammel
    ts_res = nwRegress(excessReturns, factors, 1, nLagsTS)
    beta = ts_res['bv'][1:]

    # Cross-section regression
    cs_params = pd.DataFrame()
    cs_X = sm.add_constant(beta.T)
    for iObs in range(t):
        cs_params = pd.concat([cs_params, pd.DataFrame(sm.OLS(excessReturns[iObs].T, cs_X).fit().params)], axis=1)

    # Risk prices and Fama-MacBeth standard errors and t-stats
    RiskPrices = cs_params.mean(axis=1).T
    covGamma = (cs_params.T.sub(RiskPrices).T @ cs_params.T.sub(RiskPrices)) / t ** 2
    # seGamma = sqrt((cs_params.T.sub(RiskPrices)**2).sum(axis=0)/t**2)
    seGamma = sqrt(diag(covGamma))
    tGammaFM = RiskPrices / seGamma

    # Adding a Shanken (1992) corrections as per Goyal (2012) eq. (33)
    covRiskFactors = ((factors - mean(factors, axis=0)).T @ (factors - mean(factors, axis=0))) / (t - nFactors)
    c = RiskPrices[1:] @ np.linalg.inv(covRiskFactors) @ RiskPrices[1:].T  # Excluding the constant
    covShanken = 1 / t * ((1 + c) * (t * covGamma.iloc[1:, 1:]) + covRiskFactors)
    seGammaShanken = sqrt(diag(covShanken)).T
    seGammaShanken = np.insert(seGammaShanken, 0, seGamma[0])
    tGammaShanken = RiskPrices / seGammaShanken

    # Mean and fitted excess returns
    meanReturns = pd.DataFrame(mean(excessReturns, 0))
    fittedValues = (pd.DataFrame(cs_X) @ pd.DataFrame(RiskPrices)).T

    # Cross sectional R^2
    Ones = pd.DataFrame(np.ones((1, n), dtype=int)).T
    errResid = meanReturns - fittedValues
    s2 = mean(errResid ** 2, axis=1)
    vary = mean((meanReturns.T - Ones * mean(meanReturns, axis=1)) ** 2)
    rSquared = 100 * (1 - s2 / vary)
    MAPE = mean(abs(errResid), axis=0)
    RMSE = sqrt(mean(errResid ** 2, axis=0))

    fmbOut = dict()
    fmbOut['FS_beta'] = ts_res['bv']
    fmbOut['FS_tstat'] = ts_res['tbv']
    fmbOut['FS_R2'] = ts_res['R2v']
    fmbOut['FS_R2adj'] = ts_res['R2vadj']

    fmbOut['SS_gamma'] = RiskPrices
    fmbOut['SS_seGammaFM'] = seGamma
    fmbOut['SS_seGammaShanken'] = seGammaShanken
    fmbOut['SS_tstatFM'] = tGammaFM
    fmbOut['SS_tstatShanken'] = tGammaShanken
    fmbOut['SS_r2'] = rSquared
    fmbOut['SS_fit'] = fittedValues
    fmbOut['MeanReturns'] = meanReturns
    fmbOut['MAPE'] = MAPE
    fmbOut['RMSE'] = RMSE
    fmbOut['cShanken'] = c
    return fmbOut


def nwRegress(y, x, constant, nlag):
    # Function regResults = nwRegress(y,x,constant, method,nlag)
    # Purpose:  Estimate a linear regression model Newey-West standard errors.
    #           a constant is added by default unless otherwise specified
    # Input:    y = TxN matrix of dependent variables (N seperate regressions)
    #           x = A TxK matrix of common explanatory variables
    #           constant = 1 to add constant internally, 0 otherwise
    #           nlag = scalar indicating the number of lags to include
    # Output:   A structure including:
    #           bv = A K x N matrix of parameter estimates
    #           sbv= A K x N matrix of user-selected standard errors
    #           tbv= A K x N matrix of t-statistics
    #           R2v= A N x 1 vector of r-square values
    #           R2vadj= A N x 1 vector of adjusted r-square values

    # Preliminaries
    # y = mat(y.values)
    # x = mat(x.values)
    if constant == 1:
        x = sm.add_constant(x)

    nObs, nReg = y.shape
    nVars = x.shape[1]
    OnesObs = pd.DataFrame(np.ones((1, nObs), dtype=int)).T
    OnesVars = pd.DataFrame(np.ones((1, nVars), dtype=int)).T

    # Coefficient estimates
    bv = sm.OLS(y, x).fit().params

    # Input for standard errors
    Exx = x.T @ x / nObs
    errv = y - x @ bv

    # Coefficient determination
    s2 = mean(np.square(errv), axis=0)
    vary = mat(mean((y - OnesObs @ mean(y, axis=0)) ** 2, axis=0))
    s2vary = np.divide(s2, vary)
    R2v = 100 * (1 - s2vary).T
    R2vadj = 100 * (1 - s2vary * (nObs - 1) / (nObs - nVars)).T

    # Newey-West standard errors

    # Preallocations
    sbv = np.zeros((nVars, nReg))
    tbv = np.zeros((nVars, nReg))

    # Individual regressions for each dependent variable
    for iReg in range(nReg):
        ww = 1
        err = errv[:, iReg]  # (:,iReg)
        inner = (x * (err @ OnesVars.T)).T @ (x * (err @ OnesVars.T)) / nObs

        for iLag in range(1, nlag):
            innadd = (x[1:(nObs - iLag), :] * (err[1:(nObs - iLag)] @ OnesVars.T)).T @ (
                    x[1 + iLag:nObs, :] * (err[1 + iLag:nObs] @ OnesVars.T)) / nObs
            inner = inner + (1 - ww * iLag / (nlag + 1)) * (innadd + innadd.T)

        varb = sm.OLS(inner, Exx).fit().params @ np.linalg.inv(Exx) / nObs

        # Standard errors
        sbv[:, iReg] = sqrt(diag(varb))

        # t-stats
        tbv[:, iReg] = bv[:, iReg] / sbv[:, iReg]

    # Structure for results:
    nwOut = dict()
    nwOut['bv'] = bv
    nwOut['tbv'] = tbv
    nwOut['R2v'] = R2v
    nwOut['R2vadj'] = R2vadj
    nwOut['resid'] = errv
    return nwOut


def PricingErrorPlot(dfFittedValues, dfMeanReturns):
    plt.style.use('seaborn')
    dfPlot = dfFittedValues
    dfPlot = dfPlot.append(dfMeanReturns)
    dfPlot.index = ['FV', 'MR']
    dfPlot = dfPlot.T
    plt.scatter(dfPlot['FV'], dfPlot['MR'])
    plt.plot([-1, 1], [-1, 1], color='orange')
    ymin, ymax = min(dfPlot.min(axis=0)), max(dfPlot.max(axis=0))
    # Set the y limits making the maximum 5% greater
    plt.ylim(ymin * 1.5, ymax * 1.5)
    plt.xlim(ymin * 1.5, ymax * 1.5)
    plt.grid(False)
    plt.title("Pricing errors")
    plt.xlabel("Model implied returns (in %)")
    plt.ylabel("Average realized returns (in %)")
    variablelabel = ['P4', 'P3', 'P2', 'P1']
    for i, txt in enumerate(variablelabel):
        plt.annotate(txt, (dfPlot['FV'][i], dfPlot['MR'][i]))


def sortPortfolio(returns, marketeq,  signal, aPortfolios=[0, .20, .80, 1]):
    from scipy import stats
    #returns = dfReturns
    #marketeq = dfMarketCap
    #signal = dfPositiveSentimentSignal

    # Lag signal as sorting should be done based
    returns = returns[1:]  # TXN
    marketeq = marketeq[1:]  # TXN
    signal = signal[:-1]  # TXN

    sortPort = pd.DataFrame()

    # Lag returns based on portfolio decision
    returns.index = signal.index
    marketeq.index = signal.index

    MarketCapDummy = marketeq.iloc[:, :].ge(0.1, axis=0)
    MarketCapDummy = MarketCapDummy.where(MarketCapDummy == 1, np.nan)
    signal = signal.multiply(MarketCapDummy)
    # NumActiveCoins = signal.iloc[:, :].ge(-1.1, axis=0).sum(axis=1)

    signal = signal.replace([np.inf, -np.inf], np.nan)

    rank = signal.stack(dropna=False).groupby(level=0).rank(ascending=False, method='first').unstack()


    for iObs in range(1, returns.shape[0]):
        # calculate breakpoint for bin

        lbins = [-np.inf
            , rank.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[1], interpolation='linear'),
                 rank.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[2], interpolation='linear'),
                 np.inf]

        digitized = np.digitize(rank.iloc[iObs - 1, :].dropna(), bins=lbins)

        bin_return = [returns.iloc[iObs, :].dropna()[signal.iloc[iObs - 1, :].dropna().index][digitized == i] for i in range(1, len(lbins))]
        bin_marketeq = [marketeq.iloc[iObs, :].dropna()[signal.iloc[iObs - 1, :].dropna().index][digitized == i] for i in range(1, len(lbins))]
        bin_mwreturn = [0,0,0]
        for i in range(0,len(bin_return)):
            bin_mwreturn[i] = (bin_return[i].values * bin_marketeq[i].values).sum() / bin_marketeq[i].values.sum()


        sortmean = pd.Series(bin_mwreturn)


        sortPort = pd.concat([sortPort, sortmean], axis=1)

    sortPort = sortPort.T

    sortPort.index = signal.index[:-1]
    sortPort = sortPort.rename(columns={0 : "P1", 1 : "P2", 2 : "P3"})
    sortPort['LS'] = sortPort['P3'] - sortPort['P1']

    return sortPort


# def sortPortfolio4(returns, marketeq,  signal, aPortfolios=[0, .25, .50, .75, 1]):
#     from scipy import stats
#     #returns = dfReturns
#     #marketeq = dfMarketCap
#     #signal = dfPositiveSentimentSignal
#
#     # Lag signal as sorting should be done based
#     returns = returns[1:]  # TXN
#     marketeq = marketeq[1:]  # TXN
#     signal = signal[:-1]  # TXN
#
#     sortPort = pd.DataFrame()
#
#     # Lag returns based on portfolio decision
#     returns.index = signal.index
#     marketeq.index = signal.index
#
#     MarketCapDummy = marketeq.iloc[:, :].ge(0.1, axis=0)
#     MarketCapDummy = MarketCapDummy.where(MarketCapDummy == 1, np.nan)
#     signal = signal.multiply(MarketCapDummy)
#     # NumActiveCoins = signal.iloc[:, :].ge(-1.1, axis=0).sum(axis=1)
#     signal = signal.replace([np.inf, -np.inf], np.nan)
#     rank = signal.stack(dropna=False).groupby(level=0).rank(ascending=False, method='first').unstack()
#
#
#
#     for iObs in range(1, returns.shape[0]):
#         # calculate breakpoint for bin
#
#         lbins = [-np.inf
#             , signal.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[1], interpolation='higher'),
#                  signal.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[2], interpolation='midpoint'),
#                  signal.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[3], interpolation='lower'),
#                  np.inf]
#         lbins1 = [-np.inf
#             , signal.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[1], interpolation='linear'),
#                  signal.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[2], interpolation='linear'),
#                  signal.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[3], interpolation='linear'),
#                  np.inf]
#         lbins2 = [-np.inf
#             , signal.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[1], interpolation='midpoint'),
#                  signal.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[2], interpolation='midpoint'),
#                  signal.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[3], interpolation='midpoint'),
#                  np.inf]
#
#
#         digitized = np.digitize(signal.iloc[iObs - 1, :].dropna(), bins=lbins)
#         digitized1 = np.digitize(signal.iloc[iObs - 1, :].dropna(), bins=lbins1)
#         digitized2 = np.digitize(signal.iloc[iObs - 1, :].dropna(), bins=lbins2)
#
#
#         bin_return = [returns.iloc[iObs, :].dropna()[signal.iloc[iObs - 1, :].dropna().index][digitized == i] for i in range(1, len(lbins))]
#         bin_marketeq = [marketeq.iloc[iObs, :].dropna()[signal.iloc[iObs - 1, :].dropna().index][digitized == i] for i in range(1, len(lbins))]
#         bin_mwreturn = [0,0,0,0]
#         for i in range(0,len(bin_return)):
#             bin_mwreturn[i] = (bin_return[i].values * bin_marketeq[i].values).sum() / bin_marketeq[i].values.sum()
#
#
#         sortmean = pd.Series(bin_mwreturn)
#
#
#         sortPort = pd.concat([sortPort, sortmean], axis=1)
#
#     sortPort = sortPort.T
#
#     sortPort.index = signal.index[:-1]
#     sortPort = sortPort.rename(columns={0 : "Low", 1 : "Mid1", 2 : "Mid2", 3 : "Top"})
#     sortPort['LS'] = sortPort['Top'] - sortPort['Low']
#
#     return sortPort


def sortPortfolio4(returns, marketeq, signal, aPortfolios=[0, .25, .50, .75, 1]):
    from scipy import stats
     #returns = dfReturns
     #marketeq = dfMarketCap
     #signal = dfPositiveSentimentSignal

    # Lag signal as sorting should be done based
    returns = returns[1:]  # TXN
    marketeq = marketeq[1:]  # TXN
    signal = signal[:-1]  # TXN

    sortPort = pd.DataFrame()

    # Lag returns based on portfolio decision
    returns.index = signal.index
    marketeq.index = signal.index

    MarketCapDummy = marketeq.iloc[:, :].ge(0.1, axis=0)
    MarketCapDummy = MarketCapDummy.where(MarketCapDummy == 1, np.nan)
    signal = signal.multiply(MarketCapDummy)
    # NumActiveCoins = signal.iloc[:, :].ge(-1.1, axis=0).sum(axis=1)
    signal = signal.replace([np.inf, -np.inf], np.nan)

    rank = signal.stack(dropna=False).groupby(level=0).rank(ascending=False, method='first').unstack()


    for iObs in range(1, returns.shape[0]):
        # calculate breakpoint for bin

        lbins = [-np.inf
            , rank.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[1], interpolation='linear'),
                  rank.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[2], interpolation='linear'),
                  rank.iloc[iObs - 1, :].dropna().quantile(q=aPortfolios[3], interpolation='linear'),
                  np.inf]

        digitized = np.digitize(rank.iloc[iObs - 1, :].dropna(), bins=lbins)

        bin_return = [returns.iloc[iObs, :].dropna()[signal.iloc[iObs - 1, :].dropna().index][digitized == i] for i in
                      range(1, len(lbins))]
        bin_marketeq = [marketeq.iloc[iObs, :].dropna()[signal.iloc[iObs - 1, :].dropna().index][digitized == i] for i
                        in range(1, len(lbins))]
        bin_mwreturn = [0, 0, 0, 0]
        for i in range(0, len(bin_return)):
            bin_mwreturn[i] = (bin_return[i].values * bin_marketeq[i].values).sum() / bin_marketeq[i].values.sum()

        sortmean = pd.Series(bin_mwreturn)

        sortPort = pd.concat([sortPort, sortmean], axis=1)

    sortPort = sortPort.T

    sortPort.index = signal.index[:-1]
    sortPort = sortPort.rename(columns={0: "P1", 1: "P2", 2: "P3", 3: "P4"})
    sortPort['LS'] = sortPort['P4'] - sortPort['P1']

    return sortPort