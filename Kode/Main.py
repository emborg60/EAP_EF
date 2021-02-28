import Functions
import pandas as pd
import matplotlib.pyplot as plt


def group_sentiment(dfSentiment):
    dfSentiment['datetime'] = pd.to_datetime(dfSentiment['created_utc'], unit='s')
    dfSentiment['date'] = pd.DatetimeIndex(dfSentiment['datetime']).date

    dfSentiment = dfSentiment[
        ['created_utc', 'negative_comment', 'neutral_comment', 'positive_comment', 'datetime', 'date']]

    dfSentiment = dfSentiment.groupby(by=['date']).sum()

    return dfSentiment


def cleaning(df):
    # Importing Bot user names
    bots = pd.read_csv(r'Data\Bots.csv', index_col=0, sep=';')

    # Removing bots from the data
    df = df[~df.author.isin(bots.bot_names)]

    # Removing any NA's
    df.dropna()

    # Cleaning the text data, fuld af pis i bunden der prøver hvert enkelt før de røg sammen, slet hvis du ikke er intra
    keeplist = "?.!,'_-"
    import re
    Adj_comment = pd.DataFrame(
        [re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)'
                r'[\S]*\s?|(/u/|u/)\S+|(/r/|r/)\S+|[\x00-\x1f\x7f-\xff]|[0-9]+|(&g|&l)\S+'
                r'|[^\s\w' + keeplist + ']', "", elem) for elem in df['body']], columns=['body'])

    df['body'] = Adj_comment['body']
    return df


period = ['2014', '2015_01', '2015_02', '2015_03', '2015_04', '2015_05', '2015_06', '2015_07', '2015_08', '2015_09',
          '2015_10', '2015_11', '2015_12', '2016_01', '2016_02', '2016_03', '2016_04', '2016_05', '2016_06', '2016_07',
          '2016_08', '2016_09', '2016_10',
          '2016_11', '2016_12', '2017_01', '2017_02', '2017_03', '2017_04', '2017_05', '2017_06', '2017_07', '2017_08',
          '2017_09',
          '2017_10', '2017_11', '2017_12', '2018_01', '2018_02', '2018_03', '2018_04', '2018_05', '2018_06', '2018_07',
          '2018_08',
          '2018_09', '2018_10', '2018_11', '2018_12', '2019_01', '2019_02', '2019_03', '2019_04', '2019_05', '2019_06',
          '2019_07',
          '2019_08', '2019_09']

dfAllData = pd.DataFrame()
for sPeriod in period:
    query = r"""
       #standardSQL  
        SELECT author, subreddit, created_utc, score, controversiality, body  
        FROM `fh-bigquery.reddit_comments.{}`  
        WHERE REGEXP_CONTAINS(body, r'(?i)\b Dash\b')
    """.format(sPeriod)
    dfData = Functions.collect_big_query(sQuery=query)
    print(sPeriod + ' Collected')

    print(sPeriod + ' cleaned')

    dfAllData = dfAllData.append(dfData)
    del dfData

    dfAllData.to_csv('Dash_sentiment.csv')

coin_list = ['BCH', 'Cardona', 'dogecoin', 'EOS', 'ETH', 'LTC', 'XRP', 'Monero', 'BNB', 'IOTA', 'TEZOS']

dfSubRed = pd.DataFrame()
for scoin in coin_list:
    dfTemp = pd.read_csv(scoin + '_sentiment.csv', index_col=0)
    dfTemp = dfTemp.dropna()
    dfSubRed = pd.concat([dfSubRed, pd.DataFrame(dfTemp.subreddit.value_counts()[:10].index),
                          pd.DataFrame(dfTemp.subreddit.value_counts()[:10].values)], axis=1)

# Removing disturbing subreddits:
# EOS:
EOS_list = ['ffxiv', 'photography', 'masseffect', 'whowouldwin', 'astrophotography', 'elementaryos']
dfTemp = pd.read_csv('EOS_sentiment.csv', index_col=0)
dfTemp = dfTemp[~dfTemp['subreddit'].isin(EOS_list)]
dfTemp.to_csv('EOS_R_Sentiment.csv')

# Ripple: indianapolis
XRP_list = ['indianapolis']
dfTemp = pd.read_csv('XRP_sentiment.csv', index_col=0)  # 510558
dfTemp = dfTemp[~dfTemp['subreddit'].isin(XRP_list)]
dfTemp.to_csv('XRP_R_Sentiment.csv')

# BNB: SquaredCircle, dragonballfighterz, StreetFighter, step1, AirBnB
BNB_list = ['SquaredCircle', 'dragonballfighterz', 'StreetFighter', 'step1', 'AirBnB']
dfTemp = pd.read_csv('BNB_R_Sentiment.csv', index_col=0)  # 109630
dfTemp = dfTemp[~dfTemp['subreddit'].isin(BNB_list)]
dfTemp.to_csv('BNB_R_Sentiment.csv')

# New coin list
coin_list_R = ['BCH', 'Cardona', 'dogecoin', 'EOS_R', 'ETH', 'LTC', 'XRP_R', 'Monero', 'BNB_R', 'IOTA', 'TEZOS']

# Removing NA's
for scoin in coin_list_R:
    dfTemp = pd.read_csv(scoin + '_sentiment.csv', index_col=0)
    dfTemp = dfTemp.dropna()
    dfTemp.to_csv(scoin + 'NA_Sentiment.csv')

coin_list_NA = ['BTC', 'BCHNA', 'CardonaNA', 'dogecoinNA', 'EOS_RNA', 'ETHNA', 'LTCNA', 'XRP_RNA', 'MoneroNA',
                'BNB_RNA',
                'IOTANA', 'TEZOSNA', ]

for scoin in coin_list_NA:
    dfTemp = pd.read_csv(scoin + '_Sentiment.csv', index_col=0)
    dfTemp = cleaning(dfTemp)
    # dfAllData = Functions.language_filter(dfAllData, series='body', language_select='en')
    dfTemp = dfTemp.reset_index(drop=True)
    dfTemp = Functions.get_sentiment(dfTemp, series='body')
    dfTemp = group_sentiment(dfTemp)
    dfTemp.to_csv(scoin + '_Actual_Sentiment.csv')

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

for scoin in coin_list:
    dfPositiveSentimentSignal[scoin] = (dfPositive[scoin]) / (dfPositive[scoin] + dfNeutral[scoin] + dfNegative[scoin])
    dfNegativeSentimentSignal[scoin] = (dfNegative[scoin]) / (dfPositive[scoin] + dfNeutral[scoin] + dfNegative[scoin])
    # dfAverageSentimentSignal[scoin]  = dfPositive[scoin]/ dfPositive[scoin].rolling(5).mean() - 1
    dfAveragePositiveSentimentSignal[scoin]  = dfPositive[scoin].pct_change(5)
    dfAverageNegativeSentimentSignal[scoin]  = dfNegative[scoin].pct_change(5)

dfReturns = dfReturns[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]
dfMarketCap = dfMarketCap[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]
dfPositiveSentimentSignal = dfPositiveSentimentSignal[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]
dfPositiveSentimentSignal['index'] = dfPositiveSentimentSignal.index
dfPositiveSentimentSignal = dfPositiveSentimentSignal.drop_duplicates('index', keep='first')
dfPositiveSentimentSignal = dfPositiveSentimentSignal.drop(['index'], axis=1)

dfNegativeSentimentSignal = dfNegativeSentimentSignal[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]
dfNegativeSentimentSignal['index'] = dfNegativeSentimentSignal.index
dfNegativeSentimentSignal = dfNegativeSentimentSignal.drop_duplicates('index', keep='first')
dfNegativeSentimentSignal = dfNegativeSentimentSignal.drop(['index'], axis=1)

dfAveragePositiveSentimentSignal = dfAveragePositiveSentimentSignal[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]
dfAveragePositiveSentimentSignal['index'] = dfAveragePositiveSentimentSignal.index
dfAveragePositiveSentimentSignal = dfAveragePositiveSentimentSignal.drop_duplicates('index', keep='first')
dfAveragePositiveSentimentSignal = dfAveragePositiveSentimentSignal.drop(['index'], axis=1)

dfAverageNegativeSentimentSignal = dfAverageNegativeSentimentSignal[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]
dfAverageNegativeSentimentSignal['index'] = dfAverageNegativeSentimentSignal.index
dfAverageNegativeSentimentSignal = dfAverageNegativeSentimentSignal.drop_duplicates('index', keep='first')
dfAverageNegativeSentimentSignal = dfAverageNegativeSentimentSignal.drop(['index'], axis=1)

dfMOM3 = dfMOM3[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]
dfMOM5 = dfMOM5[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]
dfMOM7 = dfMOM7[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]
dfMOM14 = dfMOM14[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]

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
dfWMR = dfWMR[datetime(2014, 1, 6).date():datetime(2019, 9, 29).date()]

# Kør hertil og så skift over til "Testing.py"


########## Table Data.1 ####

#  Panel A
dfTable = dfMarketCap
dfTable['ALL'] = dfMarketCap.mean(axis=1)
dfTable['dates'] = dfTable.index
dfTable['dates'] = pd.to_datetime(dfTable.dates)
dfEachcoin = pd.DataFrame()
for coin in coin_list:
    dfEachcoin = dfEachcoin.append(dfTable.groupby(dfTable.dates.dt.year)[coin].mean())

dfDatatable = pd.DataFrame()
dfDatatable['MC index Mean'] = round(dfTable.groupby(dfTable.dates.dt.year)['ALL'].mean() / 1000000, 2)
dfDatatable['MC index Median'] = round(dfTable.groupby(dfTable.dates.dt.year)['ALL'].median() / 1000000, 2)
dfDatatable['MC coin Mean'] = round(dfEachcoin.mean() / 1000000, 2)
dfDatatable['MC coin Median'] = round(dfEachcoin.median() / 1000000, 2)
dfDatatable.mean()

numbercoins = dfMarketCap.count(axis=1) -2

print(dfDatatable.to_latex(index=False))
dfDatatable.mean()
# Panel B
del dfDatatable

dfTable = Functions.PortfolioSort(dfReturns, dfMarketCap, dfMarketCap)
dfTable = dfTable.drop(columns='LS')
dfTable = dfTable.mean(axis=1)

dfTable.mean()
dfTable.median()
dfTable.std()
dfTable.skew()
dfTable.kurt()

dfDescriptive = pd.DataFrame()
dfBTC = dfReturns
dfDescriptive['mean'] = round(dfBTC.mean() * 100,2)
dfDescriptive['median'] = round(dfBTC.median() * 100,2)
dfDescriptive['std'] = round(dfBTC.std(),2)
dfDescriptive['skew'] = round(dfBTC.skew(),2)
dfDescriptive['kurt'] = round(dfBTC.kurt(),2)

print(dfDescriptive.to_latex())

dfPositive.sum().sum()
dfNeutral.sum().sum()
dfNegative.sum().sum()

fig1, ax1 = plt.subplots()
ax1.pie([dfPositive.sum().sum(), dfNeutral.sum().sum(), dfNegative.sum().sum()], labels = ['Positive', 'Neutral', 'Negative'], autopct='%1.1f%%', startangle=90)
