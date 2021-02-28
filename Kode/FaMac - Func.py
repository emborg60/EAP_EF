# Funktion til Fama-MacBeth 2-pass regression der skal bruges i Fama-French Framework
import numpy as np
from numpy import mat, mean, sqrt, diag
import statsmodels.api as sm
import pandas as pd

# Første forsøg kører med det downloadede Kenneth French data
returns = pd.read_csv(r'Data/5x5 FF.csv')
riskFactors = pd.read_csv(r'Data/FF3Monthly.csv')
returns = returns.drop(['Unnamed: 0'],axis=1)
returns = returns.sub(riskFactors['RF'], axis=0)
riskFactors = riskFactors.drop(['Unnamed: 0', 'RF'], axis=1)


FMB_result = FMB(returns, riskFactors, nLagsTS = 3)


def FMB(returns, riskFactors, nLagsTS):

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
    excessReturns = mat(returns.values) # Måske ikke .values

    # Shape information
    t,n = excessReturns.shape
    nFactors = factors.shape[1]

    # Time series regressions
    # X = sm.add_constant(factors) # Laver X ved at inkludere en 1 vektor på faktorer
    # ts_res = sm.OLS(excessReturns, X).fit() # First pass regression # Gammel
    ts_res  = nwRegress(excessReturns, factors, 1, nLagsTS)
    beta    = ts_res['bv'][1:]

    # Cross-section regression
    cs_params = pd.DataFrame()
    cs_X = sm.add_constant(beta.T)
    for iObs in range(t):
        cs_params = pd.concat([cs_params, pd.DataFrame(sm.OLS(excessReturns[iObs].T, cs_X).fit().params)], axis=1)

    # Risk prices and Fama-MacBeth standard errors and t-stats
    RiskPrices = cs_params.mean(axis=1).T
    covGamma   = (cs_params.T.sub(RiskPrices).T @ cs_params.T.sub(RiskPrices)) / t**2
    # seGamma = sqrt((cs_params.T.sub(RiskPrices)**2).sum(axis=0)/t**2)
    seGamma    = sqrt(diag(covGamma))
    tGammaFM = RiskPrices/seGamma

    # Adding a Shanken (1992) corrections as per Goyal (2012) eq. (33)
    covRiskFactors = ((factors - mean(factors, axis=0)).T @ (factors - mean(factors, axis=0))) / (t - nFactors)
    c              = RiskPrices[1:] @ np.linalg.inv(covRiskFactors) @ RiskPrices[1:].T # Excluding the constant
    covShanken     = 1/t * ((1+c) * (t * covGamma.iloc[1:,1:]) + covRiskFactors)
    seGammaShanken = sqrt(diag(covShanken)).T
    seGammaShanken = np.insert(seGammaShanken, 0,seGamma[0])
    tGammaShanken  = RiskPrices / seGammaShanken

    # Mean and fitted excess returns
    meanReturns = pd.DataFrame(mean(excessReturns,0))
    fittedValues = (pd.DataFrame(cs_X) @ pd.DataFrame(RiskPrices)).T

    # Cross sectional R^2
    Ones        = pd.DataFrame(np.ones((1, n), dtype=int)).T
    errResid    = meanReturns-fittedValues
    s2          = mean(errResid**2, axis=1)
    vary        = mean((meanReturns.T - Ones*mean(meanReturns,axis=1))**2)
    rSquared    = 100 * (1 - s2 / vary)
    MAPE        = mean(abs(errResid), axis=0)
    RMSE        = sqrt(mean(errResid**2, axis=0))

    fmbOut = dict()
    fmbOut['FS_beta']   = ts_res['bv']
    fmbOut['FS_tstat']  = ts_res['tbv']
    fmbOut['FS_R2']     = ts_res['R2v']
    fmbOut['FS_R2adj']  = ts_res['R2vadj']

    fmbOut['SS_gamma']          = RiskPrices
    fmbOut['SS_seGammaFM']      = seGamma
    fmbOut['SS_seGammaShanken'] = seGammaShanken
    fmbOut['SS_tstatFM']        = tGammaFM
    fmbOut['SS_tstatShanken']   = tGammaShanken
    fmbOut['SS_r2']             = rSquared
    fmbOut['SS_fit']            = fittedValues
    fmbOut['MeanReturns']       = meanReturns
    fmbOut['MAPE']              = MAPE
    fmbOut['RMSE']              = RMSE
    fmbOut['cShanken']          = c
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
    #y = mat(y.values)
    #x = mat(x.values)
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
    s2      = mean(np.square(errv), axis = 0)
    vary    = mat(mean((y - OnesObs @ mean(y, axis = 0))**2, axis = 0))
    s2vary  = np.divide(s2, vary)
    R2v     = 100 * (1 - s2vary).T
    R2vadj  = 100 * (1- s2vary * (nObs-1) / (nObs - nVars)).T


    # Newey-West standard errors

    # Preallocations
    sbv = np.zeros((nVars, nReg))
    tbv = np.zeros((nVars, nReg))

    # Individual regressions for each dependent variable
    for iReg in range(nReg):
        ww      = 1
        err     = errv[:,iReg] #  (:,iReg)
        inner   = (x * (err @ OnesVars.T)).T @ (x * (err @ OnesVars.T)) / nObs

        for iLag in range(1,nlag):
            innadd  = (x[1:(nObs-iLag),:] * (err[1:(nObs-iLag)] @ OnesVars.T)).T @ (x[1+iLag:nObs,:] * (err[1+iLag:nObs] @ OnesVars.T)) / nObs
            inner   = inner + (1 - ww * iLag/(nlag+1)) * (innadd+innadd.T)

        varb = sm.OLS(inner, Exx).fit().params  @ np.linalg.inv(Exx) / nObs
        
        # Standard errors
        sbv[:,iReg] = sqrt(diag(varb))

        # t-stats
        tbv[:,iReg] = bv[:,iReg] / sbv[:, iReg]

    # Structure for results:
    nwOut = dict()
    nwOut['bv']     = bv
    nwOut['tbv']    = tbv
    nwOut['R2v']    = R2v
    nwOut['R2vadj'] = R2vadj
    nwOut['resid']  = errv
    return nwOut



# Test med Quarterly data for at se om resultater matcher hans
sizeValuePortfolios2 = pd.read_csv(r'Data/sizeValuePortfoliosQ2.csv', sep=';' ,header=None) # fjerne kolonne 0,3 og 15
sizeValuePortfolios = pd.read_csv(r'Data/sizeValuePortfoliosQ.csv', sep=';' ,header=None, dtype=np.float64) # fjerne kolonne 0,3 og 15


famaFrenchFactorsQ = pd.read_csv(r'Data/famaFrenchFactorsQ.csv', sep=';', header=None, dtype=np.float64)
logConsGrowthQ = pd.read_csv(r'Data/logConsGrowthQ.csv', sep=';', header=None)
SMB = pd.DataFrame()
HML = pd.DataFrame()
riskFreeRate = pd.DataFrame()
SMB['SMB'] = famaFrenchFactorsQ.iloc[0:,0]
HML['HML'] = famaFrenchFactorsQ.iloc[0:,1]
logConsGrowth = logConsGrowthQ
riskFreeRate['RF'] = famaFrenchFactorsQ.iloc[0:,2]

excessReturns = sizeValuePortfolios.sub(riskFreeRate['RF'], axis=0)

meanExcess = mean(excessReturns) * 12 # Tjek

factorsTjek = pd.DataFrame()
factorsTjek['logConsGrowth'] = logConsGrowth[0]
factorsTjek['SMB'] = SMB['SMB']
factorsTjek['HML'] = HML['HML']

Test = FMB(excessReturns, factorsTjek, 12) # Præcis fucking samme resultater, don't @ me fool
