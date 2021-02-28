import pandas as pd

coin_list = ['BCH', 'Cardona', 'dogecoin', 'EOS', 'ETH', 'LTC', 'XRP', 'BNB']

dfSubRed = pd.DataFrame()
dfNA = pd.DataFrame()
for scoin in coin_list:
    dfTemp = pd.read_csv(scoin + '_sentiment.csv', index_col=0)
    dfSubRed = pd.concat([dfSubRed,pd.DataFrame(dfTemp.subreddit.value_counts()[:15].index),pd.DataFrame(dfTemp.subreddit.value_counts()[:15].values)], axis=1)


dfNA = pd.read_csv('Monero_sentiment.csv', index_col=0)
dfNA['body'].isna().sum()

# initialize sentiment classifier
sia = SIA()

# get sentiment
sentiment = dfTemp['body'].apply(sia.polarity_scores)

# create sentiment df
sentiment = pd.DataFrame(sentiment.tolist())
dfTemp = dfTemp.reset_index(drop = True)
# merge sentiment with your df

df = dfTemp.merge(sentiment, how='left', left_index=True, right_index=True) # Her den kommer fejlen!
df['sentiment'] = df['compound'].apply(categorize_sentiment)
df['sentiment'] = pd.Categorical(df['sentiment'])
binary_sentiment = df['sentiment'].str.get_dummies()

df = df.merge(binary_sentiment, how='left', left_index=True, right_index=True)