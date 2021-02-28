import pandas as pd
import datetime as dt

# Jeg slår dem lige sammen : WTF en lorte kode Chrisser....


def group_sentiment(df):
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['date'] = pd.DatetimeIndex(df['datetime']).date
    df = df[
        ['created_utc', 'negative_comment', 'neutral_comment', 'positive_comment', 'datetime', 'date']]
    df = df.groupby(by=['date']).sum()
    df = df[['negative_comment', 'neutral_comment', 'positive_comment']]
    return df


dfSentiment1 = pd.read_csv(r'Data\09to14_sentiment.csv', index_col=0)
dfSentiment1 = group_sentiment(dfSentiment1)
dfSentiment1 = dfSentiment1.sort_index()

dfSentiment2 = pd.read_csv(r'Data\15_01to17_10_sentiment.csv', index_col=0)  # Jeg får en "FutureWarning"
dfSentiment2 = group_sentiment(dfSentiment2)

dfSentiment2 = dfSentiment2[:dt.date(2017, 8, 31)]

dfReplace15_05 = pd.read_csv(r'Data\15_05_sentiment.csv', index_col=0)
dfReplace15_05 = group_sentiment(dfReplace15_05)

dfReplace17_10 = pd.read_csv(r'Data\17_10_sentiment.csv', index_col=0)
dfReplace17_10 = group_sentiment(dfReplace17_10)

dfReplace17_09 = pd.read_csv(r'Data\17_09_sentiment.csv', index_col=0)
dfReplace17_09 = group_sentiment(dfReplace17_09)


dfSentiment2 = dfSentiment2.append(dfReplace15_05)
dfSentiment2 = dfSentiment2.append(dfReplace17_10)
dfSentiment2 = dfSentiment2.append(dfReplace17_09)


dfSentiment2 = dfSentiment2.sort_index()



dfSentiment3 = pd.read_csv(r'Data\17_11to19_09_sentiment.csv', index_col=0)  # Jeg får en "FutureWarning"
dfSentiment3 = group_sentiment(dfSentiment3)



dfSentiment = pd.concat([dfSentiment1,dfSentiment2,dfSentiment3])


# Read price csv
dfPrice = pd.read_csv(r'Data\Prices_comb.csv')
del dfPrice['Unnamed: 0']

# Extract date
dfPrice['date'] = dfPrice['Date'].copy()

# Convert to datetime
dfPrice['date'] = pd.to_datetime(dfPrice['date'])
dfPrice['date'] = pd.DatetimeIndex(dfPrice['date']).date
str(dfPrice['date'][0]), dfPrice['date'][0]

# check to see if `price` and `data` have the same `date` format
print('-- string --')
print('data : ' + str(dfSentiment.index[0]))
print('price : ' + str(dfPrice['date'][0]))
print('------------------------------')
print('-- raw string --')
print('data : ' + repr(dfSentiment.index[0]))
print('price : ' + repr(dfPrice['date'][0]))

# Set date as Index
dfPrice.index = dfPrice['date']
del dfPrice['date']
# del dfPrice['Date']
# dfPrice.head()

# Combine sentiment and price data
dfData = dfPrice.merge(dfSentiment, how='left', left_index=True, right_index=True)

end_date = dfSentiment.iloc[-1].name
print(end_date)

dfData = dfData[: end_date]

del dfSentiment1
del dfSentiment2
del dfSentiment3
del dfSentiment
del dfPrice

dfData_exp = pd.DataFrame()
dfData_exp['price'] = dfData['Close']
dfData_exp['volatility'] = (dfData['High'] - dfData['Low']) / dfData['Close']
dfData_exp['volume_price'] = dfData['Volume']
dfData_exp['volume_number'] = dfData['Volume'] / dfData['Close']
dfData_exp['positive_comment'] = dfData['positive_comment']
dfData_exp['neutral_comment'] = dfData['neutral_comment']
dfData_exp['negative_comment'] = dfData['negative_comment']

dfData_exp.to_csv(r'Data\All_Merged.csv')
