import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

df = pd.read_csv(r'Data\All_Merged.csv', index_col=0)

# Missing value! Snak med chrisser hvad han har lært:
# Uf ven, sgu ikke noget der er relevant i denne sammenhæng. Man kan ikke imputere noget her, tror jeg, tænker ikke
# det virker med tidsserier

# do both of these to make stationary transformation step easier
# cant have any values nan, inf or 0 for tests to work
# fill na
df = df.fillna(1)
# fill value '0' with '1'
df = df.replace(0, 1)

# Se lige på om der standardiseres, kan ikke lige huske om det er der

def descriptive_statistics(df, series):
    stats = df[series].describe()
    print('\nDescriptive Statistics for', '\'' + series + '\'', '\n\n', stats)


def get_graphics(df, series, xlabel, ylabel, title, grid=True):
    plt.plot(pd.to_datetime(df.index), df[series])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)
    return plt.show()


# stationary tests
# unit root = statistical properties of series are not constant with time.
#
# In order to be stationary, series has to be constant with time. So if a series has a unit root, it is not stationary
#
# strict stationary = mean, variance, covariance are not function of time
# trend stationary = no root unit, but has a trend. if you remove the trend, it would be strict stationary
# difference stationary = series can be made strict stationary by differencing

# ADF Augmented Dickey Fuller Test (unit root test)
# null hypothesis = series has a unit root (a = 1)
# alt hypothesis = series has no unit root
#
# accept null = t-score is greter than critical value (there is a unit root)
# reject null = t-score is less than critical value (there is no unit root)
#
# accpet null = bad (not stationary)
# reject null = good (stationary)
#
# adf can be interpreted as a difference stationary test


def adf_test(df, series):
    results = adfuller(df[series])
    output = pd.Series(results[0:4], index=['t-score', 'p-value', '# of lags used', '# of observations'])
    for key, value in results[4].items():
        output['critical value (%s)' % key] = value
    # if t-score < critical value at 5%, the data is stationary
    # if t-score > critical value at 5%, the data is NOT stationary
    if output[0] < output[5]:
        print('\nADF: The data', '\'' + series + '\'', 'is STATIONARY \n\n', output)
    elif output[0] > output[5]:
        print('\nADF: The data', '\'' + series + '\'', 'is NOT STATIONARY \n\n', output)
    else:
        print('\nADF: There is something wrong with', '\'' + series + '\'', '\n\n', output)


# KPSS Kwiatkowski-Phillips-Schmidt-Shin Test (stationary test)
# null hypothesis = the series has a stationary trend
# alt hypothesis = the series has a unit root (series is not stationary)
#
# accept null = t-score is less than critical value (series is stationary)
# reject null = t-score is greater than the critical value (series is not stationary)
#
# accpet null = good (stationary)
# reject null = bad (not stationary)
#
# kpss classifies a series as stationary on the absence of a unit root
# (both strict stationary and trend stationary will be classified as stationary)


def kpss_test(df, series):
    results = kpss(df[series], regression='ct')
    output = pd.Series(results[0:3], index=['t-score', 'p-value', '# lags used'])
    for key, value in results[3].items():
        output['critical value (%s)' % key] = value
    # if t-score < critical value at 5%, the data is stationary
    # if t-score > critical value at 5%, the data is NOT stationary
    if output[0] < output[4]:
        print('\nKPSS: The data', '\'' + series + '\'', 'is STATIONARY \n\n', output)
    elif output[0] > output[4]:
        print('\nKPSS: The data', '\'' + series + '\'', 'is NOT STATIONARY \n\n', output)
    else:
        print('\nKPSS: There is something wrong with', '\'' + series + '\'', '\n\n', output)


# Many times, adf and kpss can give conflicting results. if so:
#
# [adf = stationary], [kpss = stationary] = series is stationary
# [adf = stationary], [kpss = NOT stationary] = series is difference stationary. use differencing to make it stationary
# [adf = NOT stationary], [kpss = stationary] = series is trend stationary. remove trend to make strict stationary
# [adf = NOT STATIONARY], [kpss = NOT STATIONARY] = series is not stationary


def series_analysis(df, series, xlabel, ylabel, title):
    # descriptive stats
    descriptive_statistics(df, series)
    # graphics
    get_graphics(df, series, xlabel, ylabel, title, grid = True)
    # stationary tests
    adf_test(df, series)
    kpss_test(df, series)


# Så går vi i gang:

# create new df for stationary data
stationary = pd.DataFrame()

# ['price']
series_analysis(df, 'price', xlabel = 'year', ylabel = 'Bitcoin Price(USD)', title = 'df[\'price\']')
# Begge siger "Not Stationary"
# ['price'] = log & diff
stationary['price'] = df['price'].apply(np.log).diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'price', xlabel = 'year', ylabel = 'Bitcoin Price(USD)', title = 'Log_Diff_Price')
# Begge siger "Stationary"

# ['volatility']
series_analysis(df, 'volatility', xlabel = 'year', ylabel = 'volatility (daily (high-low)/price)', title = 'df[\'volatility\']')
# ADF: Stationary
# KPSS: Not Stationary
# Tag differencen

# ['volatility'] = diff
stationary['volatility'] = df['volatility'].diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'volatility',  xlabel = 'year', ylabel = 'Volatility', title = 'Diff_Volatility')
# Begge siger "Stationary"

# ['volume_price']
series_analysis(df, 'volume_price', xlabel = 'year', ylabel = 'Volume_Price(USD)', title = 'df[\'volume_price\']')
# Denne graf afviger meget fra hans, men jo også på disse at vi kunne se forskellene
# i datasæt
# Begge siger: Not Stationary

# ['volume_price'] = log & diff
stationary['volume_price'] =  df['volume_price'].apply(np.log).diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'volume_price',  xlabel = 'year', ylabel = 'volume_price', title = 'Log_Diff_Volume_Price')
# Begge siger "Stationary"

# ['volume_number']
series_analysis(df, 'volume_number', xlabel = 'year', ylabel = 'Number of Bitocins exchanged', title = 'df[\'volume_number\']')
# Vores er MEGET større tal, 10 gange så store!
# Vores siger begge Not Stationary, hvor han har mikset situation

# ['volume_number'] = diff
stationary['volume_number'] = df['volume_number'].diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'volume_number',  xlabel = 'year', ylabel = 'volume_number', title = 'Diff_Volume_Number')
# Får begge stationary selv uden at tage log som vi har gjort i de andre scenarier
# hvor vi har haaft begge til Not stationary initielt. Så det er et valg vi skal tage
# Måske tjekke om de andre også bliver stationære ved bare differencen. På nær pris
# som virker til at være en god idé også at køre med log grundet relationen til growth procenter


# ['positive_comment']
series_analysis(df, 'positive_comment', xlabel = 'year', ylabel = 'Number of Positive Comments', title = 'df[\'positive_comment\']')
# Den ligner nu hvor vi har fikset data
# Men vi får stationær/ikke stationær hvor han får 2 gange ikke stationær
# Gør ikke noget

# ['positive_comment'] = log & diff
stationary['positive_comment'] = df['positive_comment'].apply(np.log).diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'positive_comment', xlabel = 'year', ylabel = 'Number of Positive Comments', title = 'Log_Diff_Positive_Comment')
# Det ligner vi stadig har en fejl i måneden 2015-05!


# ['neutral_comment']
series_analysis(df, 'neutral_comment', xlabel = 'year', ylabel = 'Number of Neutral Comments', title = 'df[\'neutral_comment\']')
# Same, problemer med 0-værdier og for store udsving og begge siger stationary

# ['neutral_comment'] = log & diff
stationary['neutral_comment'] = df['neutral_comment'].apply(np.log).diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'neutral_comment', xlabel = 'year', ylabel = 'Number of Neutral Comments', title = 'Log_Diff_Neutral_Comment')
# Same

# ['negative_comment']
series_analysis(df, 'negative_comment', xlabel = 'year', ylabel = 'Number of Negative Comments', title = 'df[\'negative_comment\']')
# Same

# ['negative_comment'] = log & diff
stationary['negative_comment'] = df['negative_comment'].apply(np.log).diff().dropna()
# run tests to see if stationary
series_analysis(stationary, 'negative_comment', xlabel = 'year', ylabel = 'Number of Negative Comments', title = 'Log_Diff_Negative_Comment')
# Same

stationary.to_csv(r'Data\stationary_data_all.csv')