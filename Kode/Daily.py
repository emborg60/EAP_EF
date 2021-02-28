# Kør herfra ved start for at få fat i de nødvendige funktioner og dataframes

import Functions
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

coin_list_NA = ['BTC', 'BCHNA', 'CardonaNA', 'dogecoinNA', 'EOS_RNA', 'ETHNA', 'LTCNA', 'XRP_RNA', 'MoneroNA',
                'BNB_RNA',
                'IOTANA', 'TEZOSNA', ]
coin_list = ['BTC', 'BCH', 'Cardona', 'dogecoin', 'EOS', 'ETH', 'LTC', 'XRP', 'Monero', 'BNB', 'IOTA', 'TEZOS', ]

dfAllCoins = pd.DataFrame()
dfWMR = pd.read_csv('Data/' + coin_list[0] + '_marketdata.csv', sep=';', thousands=',', decimal='.')
dfWMR['Date'] = pd.to_datetime(dfWMR['Date'], format='%b %d, %Y')
dfWMR['Date'] = pd.DatetimeIndex(dfWMR['Date']).date
dfWMR.index = dfWMR['Date']
dfWMR = dfWMR.sort_index()
for column in dfWMR.columns:
    dfWMR = dfWMR.drop(columns=column)

dfPrices = dfWMR
dfReturns = dfWMR
dfMarketCap = dfWMR
dfPositive = dfWMR
dfNeutral = dfWMR
dfNegative = dfWMR
dfMOM3 = dfWMR
dfMOM5 = dfWMR
dfMOM7 = dfWMR
dfMOM14 = dfWMR

for i in range(0, len(coin_list)):
    dfMarket = pd.read_csv('Data/' + coin_list[i] + '_marketdata.csv', sep=';', thousands=',', decimal='.')

    dfMarket['Date'] = pd.to_datetime(dfMarket['Date'], format='%b %d, %Y')
    dfMarket['Date'] = pd.DatetimeIndex(dfMarket['Date']).date
    dfMarket.index = dfMarket['Date']
    dfMarket = dfMarket.sort_index()
    dfMarket['Return'] = dfMarket['Close**'].pct_change()
    dfMarket = dfMarket[1:]

    dfMarket['Mom3'] = dfMarket.Return.rolling(3).sum()
    dfMarket['Mom5'] = dfMarket.Return.rolling(5).sum()
    dfMarket['Mom7'] = dfMarket.Return.rolling(7).sum()
    dfMarket['Mom14'] = dfMarket.Return.rolling(14).sum()

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfMarket['Return']
    dfReturns = dfReturns.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfMarket['Close**']
    dfPrices = dfPrices.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfMarket['Mom3']
    dfMOM3 = dfMOM3.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfMarket['Mom5']
    dfMOM5 = dfMOM5.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfMarket['Mom7']
    dfMOM7 = dfMOM7.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfMarket['Mom14']
    dfMOM14 = dfMOM14.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfMarket['Market Cap']
    dfTemp[coin_list[i]] = dfTemp[coin_list[i]].fillna(method = 'ffill')
    dfMarketCap = dfMarketCap.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfSentiment = pd.read_csv('Data/' + coin_list_NA[i] + '_Actual_Sentiment.csv', index_col=0, sep=',')
    if coin_list[i] == 'BTC':
        # dfSentiment = pd.read_csv('Data/' + coin_list_NA[i] + '_Actual_Sentiment.csv', index_col=0, sep=';')
        dfSentiment = pd.read_csv('Data/All_Merged.csv', index_col=0, sep=',')
        dfSentiment = dfSentiment[['positive_comment', 'neutral_comment', 'negative_comment']]

    dfSentiment['Date'] = dfSentiment.index
    dfSentiment['Date'] = pd.to_datetime(dfSentiment['Date'])
    dfSentiment.index = pd.DatetimeIndex(dfSentiment['Date']).date

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfSentiment['positive_comment']
    dfPositive = dfPositive.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfSentiment['negative_comment']
    dfNegative = dfNegative.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfTemp = pd.DataFrame()
    dfTemp[coin_list[i]] = dfSentiment['neutral_comment']
    dfNeutral = dfNeutral.merge(dfTemp, how='left', left_index=True, right_index=True)

    dfMarket['Coin'] = coin_list[i]
    del dfSentiment['Date']

    dfData = dfMarket.merge(dfSentiment, how='inner', left_index=True, right_index=True)
    dfData = dfData.reset_index()
    del dfData['index']

    dfAllCoins = dfAllCoins.append(dfData)

dfAllCoins = dfAllCoins.drop(['created_utc'], axis=1)

dfWMR = pd.DataFrame()
dfReturnsLag = dfReturns.iloc[1:,:]
dfMarketCapLag = dfMarketCap.iloc[:-1,:]
dfMarketCapLag.index = dfReturnsLag.index
dfWMR['WMR'] = dfReturnsLag.multiply(dfMarketCapLag).sum(axis=1) / dfMarketCapLag.sum(axis=1)

dfPositiveSentimentSignal = pd.DataFrame()
dfNegativeSentimentSignal = pd.DataFrame()
dfAveragePositiveSentimentSignal  = pd.DataFrame()
dfAverageNegativeSentimentSignal  = pd.DataFrame()
dfActiveCoin  = pd.DataFrame()

for scoin in coin_list:
    dfPositiveSentimentSignal[scoin] = (dfPositive[scoin]) / (dfPositive[scoin] + dfNeutral[scoin] + dfNegative[scoin])
    dfNegativeSentimentSignal[scoin] = (dfNegative[scoin]) / (dfPositive[scoin] + dfNeutral[scoin] + dfNegative[scoin])
    # dfAverageSentimentSignal[scoin]  = dfPositive[scoin]/ dfPositive[scoin].rolling(5).mean() - 1
    dfAveragePositiveSentimentSignal[scoin]  = dfPositive[scoin] / dfPositive[scoin].rolling(window = 5).mean()
    dfAverageNegativeSentimentSignal[scoin]  = dfNegative[scoin] / dfNegative[scoin].rolling(window = 5).mean()
    dfActiveCoin[scoin] = dfPositive[scoin] + dfNeutral[scoin] + dfNegative[scoin]

    dfAverageNegativeSentimentSignal[scoin] = dfAverageNegativeSentimentSignal[scoin].fillna(method='ffill')
    dfAveragePositiveSentimentSignal[scoin] = dfAveragePositiveSentimentSignal[scoin].fillna(method='ffill')
    #dfActiveCoin[scoin] = dfActiveCoin[scoin].fillna(0)

dfActiveCoin = dfActiveCoin[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfPrices = dfPrices[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]


dfReturns = dfReturns[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfMarketCap = dfMarketCap[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfPositiveSentimentSignal = dfPositiveSentimentSignal[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfPositiveSentimentSignal['index'] = dfPositiveSentimentSignal.index
dfPositiveSentimentSignal = dfPositiveSentimentSignal.drop_duplicates('index', keep='first')
dfPositiveSentimentSignal = dfPositiveSentimentSignal.drop(['index'], axis=1)

dfNegativeSentimentSignal = dfNegativeSentimentSignal[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfNegativeSentimentSignal['index'] = dfNegativeSentimentSignal.index
dfNegativeSentimentSignal = dfNegativeSentimentSignal.drop_duplicates('index', keep='first')
dfNegativeSentimentSignal = dfNegativeSentimentSignal.drop(['index'], axis=1)

dfAveragePositiveSentimentSignal = dfAveragePositiveSentimentSignal[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfAveragePositiveSentimentSignal['index'] = dfAveragePositiveSentimentSignal.index
dfAveragePositiveSentimentSignal = dfAveragePositiveSentimentSignal.drop_duplicates('index', keep='first')
dfAveragePositiveSentimentSignal = dfAveragePositiveSentimentSignal.drop(['index'], axis=1)

dfAverageNegativeSentimentSignal = dfAverageNegativeSentimentSignal[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfAverageNegativeSentimentSignal['index'] = dfAverageNegativeSentimentSignal.index
dfAverageNegativeSentimentSignal = dfAverageNegativeSentimentSignal.drop_duplicates('index', keep='first')
dfAverageNegativeSentimentSignal = dfAverageNegativeSentimentSignal.drop(['index'], axis=1)

dfMOM3 = dfMOM3[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfMOM5 = dfMOM5[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfMOM7 = dfMOM7[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]
dfMOM14 = dfMOM14[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]

riskFactors = pd.read_csv('Data/FF3Daily.CSV', index_col=0)
riskFactors = riskFactors.reset_index()
riskFactors = riskFactors[:-1]
riskFactors['index'] = pd.to_datetime(riskFactors['index'], format="%Y%m%d")
riskFactors.index = pd.DatetimeIndex(riskFactors['index']).date
del riskFactors['index']
riskFactors = riskFactors['RF']
dfWMR = dfWMR.merge(riskFactors, how='left', left_index=True, right_index=True)
dfWMR['RF'] = dfWMR.RF.fillna(method='ffill')
dfWMR['Mkt-RF'] = dfWMR['WMR'] - dfWMR['RF']
dfWMR = dfWMR[datetime(2014, 1,6).date():datetime(2019, 9, 29).date()]

#
# # Max = dfActiveCoin.max()
# # Min = dfActiveCoin.min()
# # Avg = dfActiveCoin.mean().round(0)
# # dfActiveCoin['TEZOS'].first_valid_index()
# dfActiveCoin.plot.box(grid='True')
#
# #Graph of Price along with amount of comments, for Bitcoin
# dfGraph = pd.DataFrame()
# dfGraph['Comments'] = (dfActiveCoin['BTC'] - dfActiveCoin['BTC'].min()) / (dfActiveCoin['BTC'].max() - dfActiveCoin['BTC'].min())
# dfGraph['Price'] = (dfPrices['BTC'] - dfPrices['BTC'].min()) / (dfPrices['BTC'].max() - dfPrices['BTC'].min())
# plt.style.use('seaborn')
# fig, ax1 = plt.subplots(1, 1, figsize=(9, 3))
# ax1.plot(dfGraph.index,
#              dfGraph['Comments'], label="Comments", color='Tomato')
# ax1.plot(dfGraph.index,
#              dfGraph['Price'], label='Price', color='Gold')
# ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., ncol=2, prop={'size': 14})
# plt.show()
