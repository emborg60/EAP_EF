import pandas as pd
import numpy as np
from langdetect import detect_langs

# import data
data = pd.read_csv(r'Data\17_09_clean.csv')

# reindex since data has been filtered out
data = data.reset_index(drop=True)


def language_filter(df, series=str, language_select=str):
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


data = language_filter(df=data, series='body', language_select='en')
# Before -> 1773664
# After ->
data.to_csv(r'Data\17_09_filtered.csv')
