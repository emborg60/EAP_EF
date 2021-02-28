import bitfinex
import time
import datetime as dt
import pandas as pd
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt

# def fetch_data(start, stop, symbol, interval, tick_limit, step):
#     # Create api instance
#     api_v2 = bitfinex.bitfinex_v2.api_v2()
#     data = []
#     start = start - step
#     while start < stop:
#         start = start + step
#         end = start + step
#         res = api_v2.candles(symbol=symbol, interval=interval,
#                              limit=tick_limit, start=start,
#                              end=end)
#         data.extend(res)
#         time.sleep(2)
#
#         names = ['time', 'open', 'close', 'high', 'low', 'volume']
#         dfdata = pd.DataFrame(data, columns=names)
#         dfdata.drop_duplicates(inplace=True)
#         dfdata['time'] = pd.to_datetime(dfdata['time'], unit='ms')
#         dfdata.set_index('time', inplace=True)
#         dfdata.sort_index(inplace=True)
#
#     return dfdata
#
#
# # Define query parameters
# pair = 'btcusd'  # Currency pair of interest
# bin_size = '1D'  # This will return minute data
# limit = 1000  # We want the maximum of 1000 data points
#
# time_step = 60000000  # Tick size this is equal to 1 minute
# # start date:
# t_start = dt.datetime(2018, 11, 10, 0, 0).date
# t_start = time.mktime(t_start.timetuple()) * 1000
#
# # end date:
# t_stop = dt.datetime(2018, 11, 16, 0, 0).date
# t_stop = time.mktime(t_stop.timetuple()) * 1000
#
# pair_data = fetch_data(start=t_start, stop=t_stop, symbol=pair,
#                        interval=bin_size, tick_limit=limit,
#                        step=time_step)

# Yahoo financials
ticker = 'BTC-USD'
financial = YahooFinancials(ticker)
dfData = pd.DataFrame.from_dict(financial.get_historical_price_data(start_date='2010-07-15', end_date='2019-09-30', time_interval='daily')["BTC-USD"]["prices"])
dfPrice = pd.read_csv(r'Data\yahoo_price.csv')

# Alternativ
# Install yfinance package.
#import yfinance as yf
# Get the data for the stock Apple by specifying the stock ticker, start date, and end date
#PriceData = yf.download('BTC-USD', '2010-07-15', '2019-09-30') # Passer sgu heller ikke helt med det vi har i forvejen, og går også kun tilbage til 14
# Jeg foreslår at vi tager alt det data vi kan fra de her pakker, og så kun holder det fra før 14 fra GitHub sættet



dfNewObs = pd.DataFrame()
dfNewObs['Date'] = dfData["formatted_date"]
dfNewObs['Open'] = dfData['open']
dfNewObs['High'] = dfData['high']
dfNewObs['Low'] = dfData['low']
dfNewObs['Close'] = dfData['close']
dfNewObs['Volume'] = dfData['volume'].values
# Går fra 16-09-2014 til 29-09-2019

end_date = dfNewObs.iloc[0]['Date']

dfPrice = dfPrice[:1523]

dfPrice_comb = dfPrice.append(dfNewObs)

#plt.plot(dfPrice_comb['Date'], dfPrice_comb['Close'])

dfPrice_comb.to_csv(r'Data\Prices_comb.csv')