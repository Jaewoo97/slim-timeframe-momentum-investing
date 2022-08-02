# Last modified: 220802

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
import argparse

def args_parser():
    parser = argparse.ArgumentParser(
        description="Slim Timeframe Momentum Investing")
    parser.add_argument(
        "--startDate",
        help="Start date for calculation. Format: 'YYYY-MM-DD'",
        required=True,
    )
    parser.add_argument(
        "--endDate",
        help="End date for calculation. Format: 'YYYY-MM-DD'",
        required=True,
    )
    parser.add_argument(
        "--market",
        help="Market type. Currently available options: KOR, US",
        default='KOR',
    )
    parser.add_argument("--numMinimization", help="Number of stocks to minimize for ")
    return parser.parse_args()

args = args_parser()
if args.market not in ['KOR', 'US']:
    raise Exception("Unsupported market. Add market symbol if needed.")
if args.market == 'KOR':
    market = ['KOSPI', 'KOSDAQ']
elif args.market == 'US':
    marekt = market = ['NASDAQ', 'NYSE']

# Parameters
num_minimization = 10    # number of stocks to iteratively construct the Markowtiz model
start_date = args.startDate
end_date = args.endDate
daysOpen = fdr.DataReader('KS11', start_date, end_date)  # To count days the market was open
buyAmount = 10000000.0
erroneousStockSymbols = []
errSymbs = pd.read_csv('errSymbols.csv')

cutoffPercentage = 0.01
numDaysSampling = 5         # Number of days for criterion calculation
numPerfEval = 3         # Number of days for performance evaluation
slopeLowThresh = [0.04, 0.04, 0.04, 0.04]
slopeHighThresh = [0.0979, 0.0979, 0.0979, 0.0979]
devLowThresh = [-0.1092, -0.1092, -0.1092, -0.1092]
devHighThresh = [-0.0403, -0.0403, -0.0403, -0.0403]
criterionMult = [1.0, 1.5, 2.0, 2.5]
portSwitchRatio = 0.5       # ratio of portfolio that wil be switched daily

# slopeLowThresh = [0.04, 0.05, 0.06]
# slopeHighThresh = [0.0979, 0.08, 0.1]
# devLowThresh = [-0.1092, -0.09, -0.1]
# devHighThresh = [-0.0403, -0.05, -0.04]

# Initialization
selected_stocks = {}
selected_stocks_filtered = {}
weights = {}
finalWeights = {}
portAmountDict = {}
momentumPerf = {str(i): np.zeros([0, 1]) for i in range(0, len(slopeLowThresh))} # Save performance of momentum portfolio for the next 5 days
performance = np.zeros([0, 2])      # expected mu, expected stdev, actual next day performance
df_close = {}

# Initialize selected stocks dictionary
for dateStartIdx in range(0, len(daysOpen.index) - numDaysSampling - numPerfEval):
    samplingEndDate = daysOpen.index[dateStartIdx + numDaysSampling - 1]
    samplingEndDateStr = datetime.strftime(samplingEndDate, '%Y-%m-%d')  # portfolio calculated by buying at this date
    selected_stocks[samplingEndDateStr] = {}
    df_close[samplingEndDateStr] = {}
    portAmountDict[samplingEndDateStr] = {}
    for portIdx in range(0, len(slopeLowThresh)):
        selected_stocks[samplingEndDateStr][str(portIdx)] = {}
        df_close[samplingEndDateStr][str(portIdx)] = pd.DataFrame

# Random sampling for search of high return stocks
print("Start retrieving symbols..")
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
print("There are " + str(len(StockList)) + " stocks... Start retrieving " + str(num_stocksToCheck-len(errSymbs)) + " random stock data")

# Fetch each stock's data for the entire duration
for sampIdx, sampKey in enumerate(SamplingList):
    print("Selecting " + str(num_minimization) + " stocks with highest returns... checking "
          + str(sampIdx) + " out of " + str(num_stocksToCheck), end='\r')
    try:
        if marketName == 'NYSE' or marketName == 'NASDAQ':
            stockInfo = pdr.get_data_yahoo(StockList[sampKey], start=samplingStartDate, end=samplingEndDate)
        else:
            if StockList[sampKey] not in errSymbs.values and StockList[sampKey] not in erroneousStockSymbols:
                stockInfo = fdr.DataReader(StockList[sampKey], start_date, end_date)
            else:
                continue
    except Exception:   # Random error
        continue
    if len(stockInfo) == 0:  # In case there is an error, zero length stock info
        continue
    stockCloseFull = pd.DataFrame({'stockInfo': stockInfo['Close']})
    # elif len(stockInfo) < numDaysSampling:
    #     debugCount[0, 2] += 1  # In case of newly introduced stocks
    #     erroneousStockSymbols.append(StockList[sampKey])
    #     continue
    # if stockClose.values[-1][0] == stockClose.values[-2][0] and stockClose.values[-2][0] == stockClose.values[-3][0]:  # 거래중지 제외
    #     debugCount[0, 3] += 1
    #     erroneousStockSymbols.append(StockList[sampKey])
    #     continue

    # Find n stocks with normalized maximum returns (mu), for every timeframe and optimize portfolio
    for dateStartIdx in range(0, len(daysOpen.index)-numDaysSampling-numPerfEval):
        mu_stds = np.zeros([len(StockList), 3])     # Stack mu, stdev
        samplingStartDate = daysOpen.index[dateStartIdx]
        samplingEndDate = daysOpen.index[dateStartIdx + numDaysSampling - 1]
        prevDate = daysOpen.index[dateStartIdx + numDaysSampling - 2]
        perfCalcEndDate = daysOpen.index[dateStartIdx + numDaysSampling + numPerfEval - 1] # Including 5 days for performance calculation
        samplingStartDateStr = datetime.strftime(samplingStartDate, '%Y-%m-%d')
        samplingEndDateStr = datetime.strftime(samplingEndDate, '%Y-%m-%d') # portfolio calculated by buying at this date
        prevDateStr = datetime.strftime(prevDate, '%Y-%m-%d')
        # print("Day " + str(dateStartIdx+1) + " of " + str(len(daysOpen.index)-numDaysSampling-numPerfEval) + " days")

        # Truncate stock's full history
        stockCloseOriginal = stockCloseFull.truncate(before=samplingStartDate, after=perfCalcEndDate)
        stockClose = stockCloseOriginal[:numDaysSampling]
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
                if dateStartIdx != 0 and StockList[sampKey] in list(selected_stocks[prevDateStr][str(portIdx)].keys()):     # encourage less portfolio switch
                    criterion = criterion * criterionMult[portIdx]
                if len(selected_stocks[samplingEndDateStr][str(portIdx)]) < num_minimization:
                    selected_stocks[samplingEndDateStr][str(portIdx)][StockList[sampKey]] = criterion
                    try:
                        len(df_close[samplingEndDateStr][str(portIdx)])
                        stockCloseOriginal.columns = [str(StockList[sampKey])]
                        df_close[samplingEndDateStr][str(portIdx)] = pd.concat([df_close[samplingEndDateStr][str(portIdx)], stockCloseOriginal], axis=1)
                    except:     # first assignment to df_close
                        stockCloseOriginal.columns = [str(StockList[sampKey])]  # Assign stock symbol
                        df_close[samplingEndDateStr][str(portIdx)] = stockCloseOriginal
                elif criterion > min(selected_stocks[samplingEndDateStr][str(portIdx)].values()):
                    deleteKey = find_key(selected_stocks[samplingEndDateStr][str(portIdx)], min(selected_stocks[samplingEndDateStr][str(portIdx)].values()))
                    del selected_stocks[samplingEndDateStr][str(portIdx)][deleteKey]     # Delete lowest return
                    del df_close[samplingEndDateStr][str(portIdx)][deleteKey]
                    selected_stocks[samplingEndDateStr][str(portIdx)][StockList[sampKey]] = criterion
                    stockCloseOriginal.columns = [str(StockList[sampKey])]
                    df_close[samplingEndDateStr][str(portIdx)] = pd.concat([df_close[samplingEndDateStr][str(portIdx)], stockCloseOriginal], axis=1)
df_closeOriginal = df_close

print("Finished fetching data. Start portfolio optimization...")
# For every timeline, optimize portfolio & save performance
for dateStartIdx in range(0, len(daysOpen.index)-numDaysSampling-numPerfEval):
    print("Iterating day " + str(dateStartIdx) + " out of " + str(len(daysOpen.index)-numDaysSampling-numPerfEval), end='\r')
    samplingStartDate = daysOpen.index[dateStartIdx]
    prevSamplingEndDate = daysOpen.index[dateStartIdx + numDaysSampling - 2]
    samplingEndDate = daysOpen.index[dateStartIdx + numDaysSampling - 1]
    perfCalcEndDate = daysOpen.index[dateStartIdx + numDaysSampling + numPerfEval - 1]  # Including 5 days for performance calculation
    samplingStartDateStr = datetime.strftime(samplingStartDate, '%Y-%m-%d')
    samplingEndDateStr = datetime.strftime(samplingEndDate, '%Y-%m-%d')  # portfolio calculated by buying at this date
    prevSamplingEndDateStr = datetime.strftime(prevSamplingEndDate, '%Y-%m-%d')
    weights[samplingEndDateStr] = {}
    finalWeights[samplingEndDateStr] = {}
    for portIdx in range(0, len(slopeLowThresh)):
        foo_close = df_closeOriginal[samplingEndDateStr][str(portIdx)]
        perfClose = foo_close.iloc[-numPerfEval:]    # Data for performance calculation
        df_close = foo_close.iloc[:numDaysSampling]  # Data for portfolio optimization
        if dateStartIdx != 0:
            foo_close_prev = df_closeOriginal[prevSamplingEndDateStr][str(portIdx)]
            perfClose_prev = foo_close_prev.iloc[-(numPerfEval-1):]
            df_close_prev = foo_close_prev.iloc[:numDaysSampling+1]

        df_logret = np.log(df_close / df_close.shift(1))
        df_logret = df_logret.dropna()
        mu_df = df_logret.mean() * len(df_logret)
        mu = mu_df.to_numpy()
        Sigma_df = df_logret.cov() * len(df_logret)
        Sigma = Sigma_df.to_numpy()
        # Minimum volatility portfolio optimization
        w_1 = cp.Variable(len(df_close.columns))
        risk_1 = cp.quad_form(w_1, Sigma)
        prob = cp.Problem(cp.Minimize(risk_1), [cp.sum(w_1) == 1, w_1 >= 0])
        prob.solve()
        weights_minVol = pd.DataFrame.from_dict([{rawStock.loc[rawStock['Symbol'] == symb]['Name'].values[0]: w_1.value[idx]
                                                  for idx, symb in enumerate(selected_stocks[samplingEndDateStr][str(portIdx)].keys())}])
        sectors = pd.DataFrame.from_dict([{idx: rawStock.loc[rawStock['Symbol'] == symb]['Industry'].values[0]
                                           for idx, symb in enumerate(selected_stocks[samplingEndDateStr][str(portIdx)].keys())}])
        symbols = pd.DataFrame.from_dict([{idx: symb for idx, symb in enumerate(selected_stocks[samplingEndDateStr][str(portIdx)].keys())}])
        weights_minVol.loc[1] = sectors.values[0]
        weights_minVol.loc[2] = symbols.values[0]
        weights[samplingEndDateStr][str(portIdx)] = weights_minVol

        portAmount = perfClose.copy()  # Sampling 끝나는 날 주식 사고 다음날부터 변동
        # Filter insignificant stocks, ratio less than cutoffPercentage
        for stock in weights[samplingEndDateStr][str(portIdx)].columns:
            if weights[samplingEndDateStr][str(portIdx)][stock].iloc[0] < cutoffPercentage:
                weights[samplingEndDateStr][str(portIdx)] = weights[samplingEndDateStr][str(portIdx)].drop([stock], axis=1)
                portAmount = portAmount.drop([rawStock.loc[rawStock['Name'] == stock]['Symbol'].values[0]], axis=1)
        weights[samplingEndDateStr][str(portIdx)].values[0] = weights[samplingEndDateStr][str(portIdx)].values[0] * 1.0 / \
                                                sum(weights[samplingEndDateStr][str(portIdx)].values[0])
        # Weight filtering by switch ratio
        if dateStartIdx == 0:
            finalWeights[samplingEndDateStr][str(portIdx)] = weights[samplingEndDateStr][str(portIdx)]
        else:
            finalWeights[samplingEndDateStr][str(portIdx)] = weights[prevSamplingEndDateStr][str(portIdx)]     # To be updated for switch ratios other than 0.5
            finalWeights[samplingEndDateStr][str(portIdx)].iloc[0] = finalWeights[samplingEndDateStr][str(portIdx)].iloc[0] * portSwitchRatio   # Initialize final weights by adding previous day portfolio
            for newDateKeyName in weights[samplingEndDateStr][str(portIdx)]:        # Add today's portfolio to the initialized portfolio
                if newDateKeyName in finalWeights[samplingEndDateStr][str(portIdx)].keys():
                    finalWeights[samplingEndDateStr][str(portIdx)][newDateKeyName].iloc[0] = \
                        finalWeights[samplingEndDateStr][str(portIdx)][newDateKeyName].iloc[0] + weights[samplingEndDateStr][str(portIdx)][newDateKeyName].iloc[0] * portSwitchRatio
                else:
                    finalWeights[samplingEndDateStr][str(portIdx)][newDateKeyName] = weights[samplingEndDateStr][str(portIdx)][newDateKeyName]
                    finalWeights[samplingEndDateStr][str(portIdx)][newDateKeyName].iloc[0] = finalWeights[samplingEndDateStr][str(portIdx)][newDateKeyName].iloc[0] * portSwitchRatio

        # Calculate portfolio prices per date
        for keyName in finalWeights[samplingEndDateStr][str(portIdx)]:
            key = rawStock.loc[rawStock['Name'] == keyName]['Symbol'].values[0]
            if key in perfClose.keys():
                portAmount.at[portAmount.index[0], key] = buyAmount * float(finalWeights[samplingEndDateStr][str(portIdx)][keyName].iloc[0])\
                                                          * float(perfClose[key].iloc[0] / df_close[key].iloc[-1])
            else:
                portAmount.at[portAmount.index[0], key] = buyAmount * float(
                    finalWeights[samplingEndDateStr][str(portIdx)][keyName].iloc[0]) \
                                                          * float(perfClose_prev[key].iloc[0] / df_close_prev[key].iloc[-1])
        portAmount['합'] = 0
        portAmount = portAmount.iloc[0]
        portAmount['합'] = sum(portAmount.values[:-1])
        portAmountDict[samplingEndDateStr][str(portIdx)] = portAmount         # Raw portfolio performance of 5 days based on sampling of n days
        foo = np.array([portAmount['합']/buyAmount])
        momentumPerf[str(portIdx)] = np.vstack([momentumPerf[str(portIdx)], foo])

# Analyze portfolio switch ratio
portSwitchRatio = np.zeros([len(finalWeights.keys())-1, len(slopeLowThresh)])    # ratio of preserved stock
fooKeys = list(finalWeights.keys())
for dateIdx in range(0, len(finalWeights.keys())-1):
    todayDate = fooKeys[dateIdx]
    nextDate = fooKeys[dateIdx + 1]
    for portIdx in range(0, len(slopeLowThresh)):
        for symbKey in list(finalWeights[todayDate][str(portIdx)].keys()):
            if symbKey in list(finalWeights[nextDate][str(portIdx)].keys()):
                portSwitchRatio[dateIdx, portIdx] = portSwitchRatio[dateIdx, portIdx] + np.amin([finalWeights[todayDate][str(portIdx)][symbKey][0], finalWeights[nextDate][str(portIdx)][symbKey][0]])

print("Finished overall performance calculation! Post-process performance data..")
duration1day = {str(i): buyAmount for i in range(0, len(slopeLowThresh))}
duration1days = np.zeros([len(momentumPerf['0'][:, 0]), len(slopeLowThresh)])  # Performance for each portfolio

# Performance analysis of different hold days
for dayIdx in range(0, len(momentumPerf['0'][:, 0])):
    for portIdx in range(0, len(slopeLowThresh)):
        duration1day[str(portIdx)] = duration1day[str(portIdx)] * momentumPerf[str(portIdx)][dayIdx, 0]
        duration1days[dayIdx, portIdx] = duration1day[str(portIdx)]

# Fetch KOSPI, KOSDAQ data
KOSPIdata = pd.DataFrame()
# KOSPIdata.index = perfClose.index
foo = list(finalWeights.keys())
KOSPIdata['KOSPI'] = fdr.DataReader('KS11', foo[0], foo[-1])['Close']
n = 1
while True:
    dayBefore = KOSPIdata.index[0] - dt.timedelta(days=n)
    try:
        dump = fdr.DataReader('KS11', dayBefore, dayBefore)['Close']
        break
    except Exception:
        n = n + 1
KOSPIdata['KOSPI'] = KOSPIdata['KOSPI'] * buyAmount / float(fdr.DataReader('KS11', dayBefore, dayBefore)['Close'])
KOSPIdata['KOSDAQ'] = fdr.DataReader('KQ11', foo[0], foo[-1])['Close']
KOSPIdata['KOSDAQ'] = KOSPIdata['KOSDAQ'] * buyAmount / float(fdr.DataReader('KQ11', dayBefore, dayBefore)['Close'])
pd_momentumPerf = {}
for portIdx in range(0, len(slopeLowThresh)):
    pd_momentumPerf[str(portIdx)] = pd.DataFrame(momentumPerf[str(portIdx)])
    pd_momentumPerf[str(portIdx)].index = KOSPIdata.index
    pd_momentumPerf[str(portIdx)].columns = ['1 day']

duration1days = pd.DataFrame(duration1days)
KOSPIdata.to_clipboard()
print("Finished.")