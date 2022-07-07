# Import default packages
import operator
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import csv
import cvxpy as cp
import datetime
from utils import *

# Parameters
num_minimization = 100    # number of stocks to iteratively construct the Markowtiz model
numChosenMinimize = 10
num_weighting = 5    # number of stocks to construct the portfolio
num_stocksToCheck = 0     # number of stocks to randomly check their return. If zero, run all.
numDaysSampling = 5
market = ['KOSPI', 'KOSDAQ']
# market = ['NASDAQ', 'NYSE']
# start_date_list = ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-04-15']
start_date = '2021-07-01'
end_date = '2022-12-31'
daysOpen1 = fdr.DataReader('KS11', start_date, end_date)  # To count days the market was open
daysOpen2 = fdr.DataReader('KQ11', start_date, end_date)  # To count days the market was open


# Random sampling for search of high return stocks
print("Start retrieving symbols")
for idx, marketName in enumerate(market):
    if idx == 0:
        rawStock = fdr.StockListing(marketName)     # Raw info of stocks
        StockList = np.array(rawStock.Symbol)       # List of symbols
    else:
        foo_rawStock = fdr.StockListing(marketName)
        foo = np.array(foo_rawStock.Symbol)
        rawStock = pd.concat([rawStock, foo_rawStock])
        StockList = np.hstack([StockList, foo])
if num_stocksToCheck == 0:
    SamplingList = random.sample(range(len(StockList)), len(StockList))
    num_stocksToCheck = len(StockList)
else:
    SamplingList = random.sample(range(len(StockList)), num_stocksToCheck)
print("There are " + str(len(StockList)) + " stocks... Start retrieving " + str(num_stocksToCheck) + " random stock data")

# Find n stocks with normalized maximum returns (mu)
extractedData = np.zeros([0, 9])     # Stack mu, std, std^0.2, criterion, slope, net deviation, next day performance
selected_stocks = {start_date: {}}
debugCount = np.zeros([1, 4])
for idx, samp_i in enumerate(SamplingList):
    if idx % 10 == 0:
        print("analyzing " + str(idx) + " out of " + str(num_stocksToCheck), end='\r')
    try:
        if marketName == 'NYSE' or marketName == 'NASDAQ':
            stockInfo = pdr.get_data_yahoo(StockList[samp_i], start=start_date, end=end_date)
        else:
            stockInfo = fdr.DataReader(StockList[samp_i], start_date, end_date)  # fdr version not working for NASDAQ
    except Exception:
        debugCount[0, 0] += 1
        continue
    if len(stockInfo) == 0:     # In case there is an error, zero length stock info
        debugCount[0, 1] += 1
        continue
    elif len(stockInfo) < 4:
        debugCount[0, 2] += 1   # In case of newly introduced stocks
        continue
    stockClose = pd.DataFrame({'stockInfo': stockInfo['Close']})
    if stockClose.values[-1][0] == stockClose.values[-2][0] and stockClose.values[-2][0] == stockClose.values[-3][0]:  # 거래중지 제외
        debugCount[0, 3] += 1
        continue
    numDaysOpen = len(stockInfo.index)
    # Repeat for different time ranges
    ret = np.log(stockClose / stockClose.shift(1))
    mu_df = ret.mean() * len(stockClose)
    mu = mu_df.to_numpy()[0]
    std_df = ret.std()
    stdOriginal = std_df.to_numpy()[0]
    std = pow(stdOriginal, (1 / 5))
    criterion = mu/std
    if std == 0.0:
        debugCount[0, 4] += 1
        continue
    for dateStartIdx in range(0, numDaysOpen-numDaysSampling):
        stockCloseCropped = stockClose.iloc[dateStartIdx:dateStartIdx+numDaysSampling]
        nextDay = stockClose.iloc[dateStartIdx+numDaysSampling]
        slope = float(stockCloseCropped.iloc[-1]/stockCloseCropped.iloc[0])
        nextDayPerf = float(stockClose.iloc[dateStartIdx+numDaysSampling]/stockClose.iloc[dateStartIdx+numDaysSampling-1])
        sumNetDev = 0
        sumNetDev2 = 0
        sumNetDev5 = 0
        for devIdx in range(1, numDaysSampling-1):
            if devIdx == numDaysSampling-2:
                sumNetDev2 += 2*float((stockCloseCropped.iloc[devIdx] - stockCloseCropped.iloc[0] - devIdx*(stockCloseCropped.iloc[-1]
                         - stockCloseCropped.iloc[0])/float(numDaysSampling-1)) / stockCloseCropped.iloc[0])
                sumNetDev5 += 5*float((stockCloseCropped.iloc[devIdx] - stockCloseCropped.iloc[0] - devIdx*(stockCloseCropped.iloc[-1]
                         - stockCloseCropped.iloc[0])/float(numDaysSampling-1)) / stockCloseCropped.iloc[0])
            else:
                sumNetDev2 += float(
                    (stockCloseCropped.iloc[devIdx] - stockCloseCropped.iloc[0] - devIdx * (stockCloseCropped.iloc[-1]
                                                                                            - stockCloseCropped.iloc[
                                                                                                0]) / float(
                        numDaysSampling - 1)) / stockCloseCropped.iloc[0])
                sumNetDev5 += float(
                    (stockCloseCropped.iloc[devIdx] - stockCloseCropped.iloc[0] - devIdx * (stockCloseCropped.iloc[-1]
                                                                                            - stockCloseCropped.iloc[
                                                                                                0]) / float(
                        numDaysSampling - 1)) / stockCloseCropped.iloc[0])
            sumNetDev += float((stockCloseCropped.iloc[devIdx] - stockCloseCropped.iloc[0] - devIdx*(stockCloseCropped.iloc[-1]
                         - stockCloseCropped.iloc[0])/float(numDaysSampling-1)) / stockCloseCropped.iloc[0])
        fooExtracted = np.array([mu, stdOriginal, std, criterion, slope, sumNetDev, sumNetDev2, sumNetDev5, nextDayPerf])
        extractedData = np.vstack([extractedData, fooExtracted])
        # if len(selected_stocks[start_date]) < num_minimization:
        #     selected_stocks[start_date][StockList[samp_i]] = criterion
        # elif criterion > min(selected_stocks[start_date].values()):
        #     del selected_stocks[start_date][find_key(selected_stocks[start_date], min(selected_stocks[start_date].values()))]     # Delete lowest return
        #     selected_stocks[start_date][StockList[samp_i]] = criterion
extractedData = pd.DataFrame(extractedData, columns=['Mu', 'std', 'std^0.2', 'criterion', 'slope', 'sum deviation', 'sum deviation x2', 'sum deviation x3', 'next day perf.'])
extractedData.to_csv('momentumAnalysisData_0704.csv', encoding="utf-8-sig")
print("Finished calculation!")
#
# print("Finished selecting stocks. Check validity of selected stocks...")
# # for rep_ddate in start_date_list:
# print("Checking validity of list with start date of "+str(start_date))
# for idx, key in enumerate(selected_stocks[start_date]):
#     try:
#         if marketName == 'NYSE' or marketName == 'NASDAQ':
#             stockInfo = pdr.get_data_yahoo(key, start=start_date, end=end_date)
#         else:
#             stockInfo = fdr.DataReader(key, start_date, end_date)  # fdr version not working for NASDAQ
#         # print(str(key)+" fetched properly!")
#     except:
#         print(str(key)+" not fetched properly... idx: " + str(idx+1) + " out of " + str(len(selected_stocks[rep_date])))
#
# print("Finished checking validity of all stocks. Sorting "+str(numChosenMinimize)+" out of "+str(num_minimization))
# sorted_selected_stocks = sorted(selected_stocks.items(), key=operator.itemgetter(1))
#
# print("Finished checking validity of stocks. Calculate mu, sigma and optimal weights for each timeframe...")
# weights = {start_date: {} for start_date in start_date_list}
# for rep_date in start_date_list:
#     df_close = pd.DataFrame()
#     if market[0] == 'NYSE' or market[0] == 'NASDAQ':
#         df_close = pd.DataFrame({rawStock.loc[rawStock['Symbol'] == key]['Name'].values[0]: pdr.
#                                 get_data_yahoo(key, start=rep_date, end=end_date)['Close'] for key in selected_stocks[rep_date]})
#     else:
#         df_close = pd.DataFrame({rawStock.loc[rawStock['Symbol'] == key]['Name'].values[0]: fdr.
#                                 DataReader(key, rep_date, end_date)['Close'] for key in selected_stocks[rep_date]})
#     # Making a dataframe for log return
#     df_logret = np.log(df_close / df_close.shift(1))
#     df_logret = df_logret.dropna()
#     mu_df = df_logret.mean() * len(df_logret)
#     mu = mu_df.to_numpy()
#     Sigma_df = df_logret.cov() * len(df_logret)
#     Sigma = Sigma_df.to_numpy()
#
#     print("Time frame: "+str(rep_date)+". Calculate min vol. portfolio using " + str(len(df_close.columns)) + " assets...")
#     # Minimum volatility portfolio optmization
#     w_1 = cp.Variable(len(df_close.columns))
#     risk_1 = cp.quad_form(w_1, Sigma)
#     prob = cp.Problem(cp.Minimize(risk_1), [cp.sum(w_1) == 1, w_1 >= 0])
#     prob.solve()
#     weights_minVol = pd.DataFrame.from_dict([{rawStock.loc[rawStock['Symbol'] == symb]['Name'].values[0]: w_1.value[idx]
#                                               for idx, symb in enumerate(selected_stocks[rep_date].keys())}])
#     sectors = pd.DataFrame.from_dict([{idx: rawStock.loc[rawStock['Symbol'] == symb]['Industry'].values[0]
#                                        for idx, symb in enumerate(selected_stocks[rep_date].keys())}])
#     symbols = pd.DataFrame.from_dict([{idx: symb for idx, symb in enumerate(selected_stocks[rep_date].keys())}])
#     weights_minVol.loc[1] = sectors.values[0]
#     weights_minVol.loc[2] = symbols.values[0]
#     weights[rep_date] = weights_minVol
#     print('> Expected return of the portfolio is ', round(mu.dot(w_1.value), 4))
#     print('> Standard deviation of the portfolio is ', round(np.sqrt(w_1.value.dot(Sigma).dot(w_1.value)), 4))
#
# # Sort into one portfolio
# weighting_list = [1.0]
# sumWeights = weights[min(start_date_list)]
# sumWeights.iloc[0] = sumWeights.iloc[0] * weighting_list[0]
# for idx, rep_date in enumerate(start_date_list[1:]):
#     for stock in weights[rep_date]:
#         if stock in sumWeights.keys():
#             sumWeights[stock].iloc[0] = sumWeights[stock].iloc[0] + \
#                                         weights[rep_date][stock].iloc[0]*weighting_list[idx+1]
#         else:   # Not in portfolio, add the stock
#             sumWeights[stock] = weights[rep_date][stock]
#             sumWeights[stock].iloc[0] = sumWeights[stock].iloc[0] * weighting_list[idx+1]
# sumWeights.iloc[0] = sumWeights.iloc[0] / sum(weighting_list)
# sumWeights.to_clipboard()
#
# refineSymbols(sumWeights)       # To properly save as csv
# sumWeights.to_csv('summedPortfolio.csv', encoding="utf-8-sig")
#
# # CAPM?
# # deep learning?