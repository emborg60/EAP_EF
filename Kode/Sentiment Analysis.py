import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA

data = pd.read_csv(r'Data\17_09_filtered.csv', index_col=0)


def get_sentiment(df, series=str):
    # initialize sentiment classifier
    sia = SIA()

    # get sentiment
    sentiment = df[series].apply(sia.polarity_scores)

    # create sentiment df
    sentiment = pd.DataFrame(sentiment.tolist())

    # merge sentiment with your df
    return df.merge(sentiment, how='left', left_index=True, right_index=True)


# run function
data = get_sentiment(df=data, series='body')
data_save = data

# Bør vi se på egne cutoffs?

def categorize_sentiment(x):
    if x >= 0.05:
        return 'positive_comment'
    elif 0.05 > x > -0.05:
        return 'neutral_comment'
    elif -0.05 >= x:
        return 'negative_comment'


data['sentiment'] = data['compound'].apply(categorize_sentiment)

# convert ['sentiment'] to categorical data type
data['sentiment'] = pd.Categorical(data['sentiment'])

binary_sentiment = data['sentiment'].str.get_dummies()

data = data.merge(binary_sentiment, how='left', left_index=True, right_index=True)

data.to_csv(r'Data\17_09_sentiment.csv')

