# Kør Main først for at hente de nødvendige dataframes
# Portfolios siger top - low til at lave LS. Så positiv skal have orderes asc så dem med flest kommer i top. Modsat for negativ.
# Først, lave portfølje til at teste sentiment returns
import Functions
import pandas as pd

dfPositiveSentimentPortfolios = Functions.sortPortfolio4(dfReturns, dfMarketCap, dfPositiveSentimentSignal)
dfNegativeSentimentPortfolios = Functions.sortPortfolio4(dfReturns, dfMarketCap, dfNegativeSentimentSignal)
dfAvgPositiveSentimentPortfolios = Functions.sortPortfolio4(dfReturns, dfMarketCap, dfAveragePositiveSentimentSignal)
dfAvgNegativeSentimentPortfolios = Functions.sortPortfolio4(dfReturns, dfMarketCap, dfAverageNegativeSentimentSignal)

dfPositiveSentimentPortfolios = dfPositiveSentimentPortfolios.dropna()
dfNegativeSentimentPortfolios = dfNegativeSentimentPortfolios.dropna()
dfAvgPositiveSentimentPortfolios = dfAvgPositiveSentimentPortfolios.dropna()
dfAvgNegativeSentimentPortfolios = dfAvgNegativeSentimentPortfolios.dropna()

# Reg på konstant
dfPositiveSignificance = Functions.ReturnSignificance(dfPositiveSentimentPortfolios)  # ikke signifikant
dfNegativeSignificance = Functions.ReturnSignificance(dfNegativeSentimentPortfolios)  # ikke signifikant
dfAvgPositiveSignificance = Functions.ReturnSignificance(dfAvgPositiveSentimentPortfolios)  # signifikant
dfAvgNegativeSignificance = Functions.ReturnSignificance(dfAvgNegativeSentimentPortfolios)  # signifikant

print(round(dfAvgNegativeSignificance,4).to_latex())

# Først, lave portfølje til at teste sentiment returns
# Laver Market cap weighted returns for portfolios med percentile breakpoints [0, .40, .60, 1] (kan ændres)
# Mest nederen kode jeg har skrevet i mit fucking liv. Giver nogle fede resultater både for weekly og daily

# Compute as excess returns, i tvivl om dette skal gøres før ovenstående tests!
dfSentimentPortfolios = dfAvgPositiveSentimentPortfolios.sub(dfWMR['RF'], axis=0)
dfSentimentPortfolios = dfSentimentPortfolios.fillna(method='ffill')

# Creating Risk Factors
dfMKT = pd.DataFrame(dfWMR['Mkt-RF'])
# igen top - low. så SMB skal have mindste i top altså dsc order. mom skal være acending da winners i top.       ½

dfSMB = Functions.sortPortfolio(dfReturns, dfMarketCap, dfMarketCap, aPortfolios=[0, .30, .7, 1])
dfMOM3_Factor = Functions.sortPortfolio(dfReturns, dfMarketCap, dfMOM3, aPortfolios=[0, .30, .7, 1])
dfMOM5_Factor = Functions.sortPortfolio(dfReturns, dfMarketCap, dfMOM5, aPortfolios=[0, .30, .7, 1])
dfMOM7_Factor = Functions.sortPortfolio(dfReturns, dfMarketCap, dfMOM7, aPortfolios=[0, .30, .7, 1])
dfMOM14_Factor = Functions.sortPortfolio(dfReturns, dfMarketCap, dfMOM14, aPortfolios=[0, .30, .7, 1])
# Sorting har ascending = FALSE, så største i low, og mindste i top. LS = top - low -> SMB skal ikke ganges med -1, men det skal momentum.
RiskFactors = pd.DataFrame()
RiskFactors['MKT'] = dfMKT['Mkt-RF']
RiskFactors['SMB'] = dfSMB['LS']
RiskFactors['MOM'] = dfMOM7_Factor['LS'].multiply(-1)
RiskFactors['SENT'] = dfAvgPositiveSentimentPortfolios['LS'].multiply(-1)

RiskFactors = RiskFactors.fillna(method='ffill')

# # FInd mest signifikant MOM
# dfMOM3_Sig = Functions.ReturnSignificance2(dfMOM3_Factor.fillna(method='ffill'))
# dfMOM5_Sig = Functions.ReturnSignificance2(dfMOM5_Factor.fillna(method='ffill'))
# dfMOM7_Sig = Functions.ReturnSignificance2(dfMOM7_Factor.fillna(method='ffill'))
#
#
# print(round(dfMOM3_Sig,4).to_latex())
# print(round(dfMOM5_Sig,4).to_latex())
# print(round(dfMOM7_Sig,4).to_latex())

RiskFactors_Mkt = RiskFactors.drop(['SMB', 'MOM', 'SENT'], axis=1)
RiskFactors_Size = RiskFactors.drop(['MKT', 'MOM', 'SENT'], axis=1)
RiskFactors_Mom = RiskFactors.drop(['MKT', 'SMB', 'SENT'], axis=1)
RiskFactors_MktSize = RiskFactors.drop(['MOM', 'SENT'], axis=1)
RiskFactors_MktMom = RiskFactors.drop(['SMB', 'SENT'], axis=1)
RiskFactors_SizeMom = RiskFactors.drop(['MKT', 'SENT'], axis=1)

RiskFactors_SentMkt = RiskFactors.drop(['SMB', 'MOM'], axis=1)
RiskFactors_SentSize = RiskFactors.drop(['MKT', 'MOM'], axis=1)
RiskFactors_SentMom = RiskFactors.drop(['MKT', 'SMB'], axis=1)

# Running the Fama-MacBeth Shanken corrected two-pass cross-sectional regression
# Jeg holder øje med hvor stor second stage R2 er, da jeg er utilfreds med at den er 1
# Det er sgu nok mest First-stage r2 der betyder noget, så det er måske ok

import math
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot

nLag = RiskFactors.shape[0]
nLagTS = math.floor(4 * (nLag / 100) ** (2 / 9))
nLagTS = math.floor(nLag ** (1 / 4))
# test = dfReturns.fillna(0)
# test = test.sub(dfWMR['RF'], axis=0)
# TEST = Functions.FMB_Shank(returns=test.iloc[:-2, :], riskFactors=RiskFactors.iloc[:-2, :], nLagsTS=nLagTS)

All = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4], riskFactors=RiskFactors.iloc[:-2, :],
                          nLagsTS=nLagTS)  # R2 = 100.0
CAPM = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4], riskFactors=RiskFactors_Mkt.iloc[:-2, :],
                           nLagsTS=nLagTS)
Size = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4], riskFactors=RiskFactors_Size.iloc[:-2, :],
                           nLagsTS=nLagTS)  # R2 = 87.87
Momentum = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4], riskFactors=RiskFactors_Mom.iloc[:-2, :],
                               nLagsTS=nLagTS)  # R2 = 99.5
CAPM_Size = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4],
                                riskFactors=RiskFactors_MktSize.iloc[:-2, :], nLagsTS=nLagTS)  # R2 = 96.2
CAPM_Mom = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4],
                               riskFactors=RiskFactors_MktMom.iloc[:-2, :], nLagsTS=nLagTS)  # R2 = 99.99
Size_Mom = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4],
                               riskFactors=RiskFactors_SizeMom.iloc[:-2, :], nLagsTS=nLagTS)  # R2 = 99.54

CAPM_Sent = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4],
                                riskFactors=RiskFactors_SentMkt.iloc[:-2, :], nLagsTS=nLagTS)
Sent_Mom = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4],
                               riskFactors=RiskFactors_SentMom.iloc[:-2, :], nLagsTS=nLagTS)
Sent_Size = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4],
                                riskFactors=RiskFactors_SentSize.iloc[:-2, :], nLagsTS=nLagTS)

# CAPM = Functions.FMB_Shank(returns=dfSentimentPortfolios.iloc[:-2, 0:4], riskFactors=RiskFactors.iloc[:-2, :],
# nLagsTS=nLagTS)

Functions.PricingErrorPlot(CAPM['SS_fit'], CAPM['MeanReturns'])
Functions.PricingErrorPlot(Size['SS_fit'], Size['MeanReturns'])
Functions.PricingErrorPlot(Momentum['SS_fit'], Momentum['MeanReturns'])

Functions.PricingErrorPlot(CAPM_Size['SS_fit'], CAPM_Size['MeanReturns'])
Functions.PricingErrorPlot(CAPM_Mom['SS_fit'], CAPM_Mom['MeanReturns'])
Functions.PricingErrorPlot(Size_Mom['SS_fit'], Size_Mom['MeanReturns'])

Functions.PricingErrorPlot(Sent_Size['SS_fit'], Sent_Size['MeanReturns'])
Functions.PricingErrorPlot(Sent_Mom['SS_fit'], Sent_Mom['MeanReturns'])
Functions.PricingErrorPlot(CAPM_Sent['SS_fit'], CAPM_Sent['MeanReturns'])

# PricingErrorPlot(CAPM['SS_fit'], CAPM['MeanReturns'])
aPrint = Sent_Size
print(pd.DataFrame(aPrint['FS_beta'].round(4)).to_latex())
print(pd.DataFrame(aPrint['FS_tstat'].round(2)).to_latex())
print(pd.DataFrame(aPrint['FS_R2'].round(2)).to_latex())

print(pd.DataFrame(aPrint['SS_gamma'].round(5)).to_latex())
print(pd.DataFrame(aPrint['SS_seGammaFM'].round(3)).to_latex())
print(pd.DataFrame(aPrint['SS_seGammaShanken'].round(3)).to_latex())
print(pd.DataFrame(aPrint['SS_tstatFM'].round(4)).to_latex())
print(pd.DataFrame(aPrint['SS_tstatShanken'].round(4)).to_latex())
print(pd.DataFrame(aPrint['SS_r2'].round(2)).to_latex())


### Return significance table:

print(round(dfPositiveSignificance, 4).to_latex())
import datetime as dt
## Cummulative net excess returns for strategies:
# dfCumsum = pd.DataFrame()
#
# # Efter som alle strategier giver negativ afkast. Er vores portfolios defineret ordenligt????
# dfCumsum['SMB'] = RiskFactors['SMB'].cumsum()  # Short small long big
# dfCumsum['MOM'] = RiskFactors['MOM'].cumsum()
# dfCumsum['Sh. Pos'] = dfPositiveSentimentPortfolios['LS'].cumsum() * - 1
# dfCumsum['Sh. Neg'] = dfNegativeSentimentPortfolios['LS'].cumsum()
# dfCumsum['Mom. sh. Pos'] = dfAvgPositiveSentimentPortfolios['LS'].cumsum() * -1
# dfCumsum['Mom. sh. Neg '] = dfAvgNegativeSentimentPortfolios['LS'].cumsum() * -1
# dfCumsum.index = RiskFactors.index
# dfCumsum = dfCumsum[:-2]
#
# import matplotlib.pyplot as plt
# plt.style.use('seaborn')
# plt.plot(dfCumsum)
# plt.ylabel("Cummulative net excess returns (in %)")
# plt.legend(list(dfCumsum.columns.values))
#
# pltcum = dfCumsum.plot()
# pltcum.set_ylabel("Cummulative net excess returns (in %)")
# pltcum.legend()
# plt.legend(list(dfCumsum.columns.values))


# Boxplot til appendix
import pandas as pd
import numpy as np
# dfActiveCoin.plot.box(grid='True') # Grim..
dfData = dfAvgNegativeSentimentPortfolios

# dfPositiveSentimentPortfolios = dfPositiveSentimentPortfolios.dropna()
# dfNegativeSentimentPortfolios = dfNegativeSentimentPortfolios.dropna()
# dfAvgPositiveSentimentPortfolios = dfAvgPositiveSentimentPortfolios.dropna()
# dfAvgNegativeSentimentPortfolios = dfAvgNegativeSentimentPortfolios.dropna()
dfDescriptive = pd.DataFrame()
dfDescriptive['Mean'] = round(dfData.mean() * 100,2)
dfDescriptive['Std'] = round(dfData.std(),2)
dfDescriptive['Skewness'] = round(dfData.skew(),2)
dfDescriptive['Kurtosis'] = round(dfData.kurt(),2)

dfDescriptive = dfDescriptive.transpose()
print(dfDescriptive.to_latex())
#dfDescriptive['Sharpe'] = round((dfData.mean() - dfWMR['RF'])/2)