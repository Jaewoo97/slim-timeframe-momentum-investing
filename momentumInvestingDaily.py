# Import default packages
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pandas_datareader import data as pdr
import yfinance as yf
import random
import csv
import cvxpy as cp
from datetime import datetime
import datetime as dt
from utils import *

# Parameters
num_minimization = 10    # number of stocks to iteratively construct the Markowtiz model
market = ['KOSPI', 'KOSDAQ']
# market = ['NASDAQ', 'NYSE']
# start_date_list = ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-04-15']
start_date = '2022-01-01'
today_date = '2022-08-03'
daysOpen = fdr.DataReader('KS11', start_date, today_date)  # To count days the market was open
buyAmount = 10000000.0
erroneousStockSymbols = []
errSymbs = pd.read_csv('errSymbols.csv')

cutoffPercentage = 0.01
numDaysSampling = 5         # Number of days for criterion calculation
slopeLowThresh = [0.04, 0.04, 0.04, 0.04]
slopeHighThresh = [0.0979, 0.0979, 0.0979, 0.0979]
devLowThresh = [-0.1092, -0.1092, -0.1092, -0.1092]
devHighThresh = [-0.0403, -0.0403, -0.0403, -0.0403]
criterionMult = [1.0, 1.5, 2.0, 2.5]

# Initialization
selected_stocks = {}
weights = {}
df_close = {}

# Initialize selected stocks dictionary
sampStartDateStr = datetime.strftime(daysOpen.index[len(daysOpen)-5], '%Y-%m-%d')
sampEndDateStr = today_date
for portIdx in range(0, len(slopeLowThresh)):
    selected_stocks[str(portIdx)] = {}
    df_close[str(portIdx)] = pd.DataFrame

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
SamplingList = random.sample(range(len(StockList)), len(StockList))
num_stocksToCheck = len(StockList)
print("There are " + str(len(StockList)) + " stocks... Start retrieving " + str(num_stocksToCheck) + " random stock data")

# Fetch each stock's data for the entire duration
for sampIdx, sampKey in enumerate(SamplingList):
    print("Selecting " + str(num_minimization) + " stocks with highest returns... checking "
          + str(sampIdx) + " out of " + str(num_stocksToCheck), end='\r')
    try:
        if marketName == 'NYSE' or marketName == 'NASDAQ':
            stockInfo = pdr.get_data_yahoo(StockList[sampKey], start=sampStartDateStr, end=sampEndDateStr)
        else:
            if StockList[sampKey] not in errSymbs.values and StockList[sampKey] not in erroneousStockSymbols:
                stockInfo = fdr.DataReader(StockList[sampKey], sampStartDateStr, sampEndDateStr)
            else:
                continue
    except Exception:   # Random error
        continue
    if len(stockInfo) == 0:  # In case there is an error, zero length stock info
        continue
    stockClose = pd.DataFrame({'stockInfo': stockInfo['Close']})
    if stockClose.values[-1][0] == stockClose.values[-2][0] and stockClose.values[-2][0] == stockClose.values[-3][0]:  # 거래중지 제외
        continue
    if len(stockClose) < numDaysSampling:   # If number of sampling days not sufficient, skip
        continue

    # Calculate slope, deviation
    slope = float(stockClose.iloc[-1] / stockClose.iloc[0])
    sumNetDev = 0
    for devIdx in range(1, numDaysSampling - 1):
        sumNetDev += float((stockClose.iloc[devIdx] - stockClose.iloc[0] - devIdx * (stockClose.iloc[-1] -
                    stockClose.iloc[0]) / float(numDaysSampling - 1)) / stockClose.iloc[0])
    # Repeat for different time ranges
    ret = np.log(stockClose / stockClose.shift(1))
    mu_df = ret.mean() * len(stockClose)
    mu = mu_df.to_numpy()[0]
    std_df = ret.std()
    stdOriginal = std_df.to_numpy()[0]
    std = pow(stdOriginal, (1/5))
    if std == 0.0:
        continue
    criterion = mu/std
    # Repeat portfolio by variables
    for portIdx in range(0, len(slopeLowThresh)):
        if np.log(slope) > slopeLowThresh[portIdx] and np.log(slope) < slopeHighThresh[portIdx] and sumNetDev > devLowThresh[portIdx] and sumNetDev < devHighThresh[portIdx]:
            if len(selected_stocks[str(portIdx)]) < num_minimization:
                selected_stocks[str(portIdx)][StockList[sampKey]] = criterion
                try:
                    len(df_close[str(portIdx)])
                    stockClose.columns = [str(StockList[sampKey])]
                    df_close[str(portIdx)] = pd.concat([df_close[str(portIdx)], stockClose], axis=1)
                except:     # first assignment to df_close
                    stockClose.columns = [str(StockList[sampKey])]  # Assign stock symbol
                    df_close[str(portIdx)] = stockClose
            elif criterion > min(selected_stocks[str(portIdx)].values()):
                deleteKey = find_key(selected_stocks[str(portIdx)], min(selected_stocks[str(portIdx)].values()))
                del selected_stocks[str(portIdx)][deleteKey]     # Delete lowest return
                del df_close[str(portIdx)][deleteKey]
                selected_stocks[str(portIdx)][StockList[sampKey]] = criterion
                stockClose.columns = [str(StockList[sampKey])]
                df_close[str(portIdx)] = pd.concat([df_close[str(portIdx)], stockClose], axis=1)

print("Finished fetching data. Start portfolio optimization...")
weights = {}
for portIdx in range(0, len(slopeLowThresh)):
    weights[str(portIdx)] = {}
    df_logret = np.log(df_close[str(portIdx)] / df_close[str(portIdx)].shift(1))
    df_logret = df_logret.dropna()
    mu_df = df_logret.mean() * len(df_logret)
    mu = mu_df.to_numpy()
    Sigma_df = df_logret.cov() * len(df_logret)
    Sigma = Sigma_df.to_numpy()
    # Minimum volatility portfolio optimization
    w_1 = cp.Variable(len(df_close[str(portIdx)].columns))
    risk_1 = cp.quad_form(w_1, Sigma)
    prob = cp.Problem(cp.Minimize(risk_1), [cp.sum(w_1) == 1, w_1 >= 0])
    prob.solve()
    weights_minVol = pd.DataFrame.from_dict([{rawStock.loc[rawStock['Symbol'] == symb]['Name'].values[0]: w_1.value[idx]
                                              for idx, symb in enumerate(selected_stocks[str(portIdx)].keys())}])
    sectors = pd.DataFrame.from_dict([{idx: rawStock.loc[rawStock['Symbol'] == symb]['Industry'].values[0]
                                       for idx, symb in enumerate(selected_stocks[str(portIdx)].keys())}])
    symbols = pd.DataFrame.from_dict([{idx: symb for idx, symb in enumerate(selected_stocks[str(portIdx)].keys())}])
    weights_minVol.loc[1] = sectors.values[0]
    weights_minVol.loc[2] = symbols.values[0]
    weights[str(portIdx)] = weights_minVol

    # Filter insignificant stocks, ratio less than cutoffPercentage
    for stock in weights[str(portIdx)].columns:
        if weights[str(portIdx)][stock].iloc[0] < cutoffPercentage:
            weights[str(portIdx)] = weights[str(portIdx)].drop([stock], axis=1)
    weights[str(portIdx)].values[0] = weights[str(portIdx)].values[0] * 1.0 / \
                                            sum(weights[str(portIdx)].values[0])

print("Finished calculating weights.")
print("")