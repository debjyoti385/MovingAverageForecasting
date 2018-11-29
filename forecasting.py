############################################################################
############################################################################
# 
# This work is inspired  from 
# Citation:
# PEREIRA, V. (2018). Project: Forecasting-01-Moving_Averages, File: Python-Forecasting-01-Moving Averages.py, GitHub repository: <https://github.com/Valdecy/Forecasting-01-Moving Averages>

# Installing Required Libraries
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import mean_squared_error
from math import sqrt


################     Part 1 - Moving Averages    #############################

# Function: WMA
def weighted_moving_average(timeseries, w = [0.2, 0.3, 0.5], graph = True, horizon = 0, limit =5):

    name = 'Weighted Moving Average'
    timeseries = pd.DataFrame(timeseries.values, index = timeseries.index, columns = [timeseries.name])/1.0
    wma = pd.DataFrame(np.nan, index = timeseries.index, columns = [name])
    center = int(len(w)/2)
    start  = 0

    for i in range(center, len(timeseries) - center):
        wma.iloc[i, 0] = 0
        for j in range(0, len(w)):
            wma.iloc[i, 0]  = float(wma.iloc[i, 0] + timeseries.iloc[start + j,:]*w[j])
        start  = start  + 1

    last = wma.iloc[(start - 1),0]

    if horizon > 0:
        if len(timeseries) - center + horizon <= len(timeseries):
            time_horizon = len(timeseries)
        else:
            time_horizon = len(timeseries) - center + horizon
        time_horizon_index = pd.date_range(timeseries.index[0], periods = time_horizon, freq = timeseries.index.inferred_freq)
        pred = pd.DataFrame(np.nan, index = time_horizon_index, columns = ["Prediction"])
        for i in range(0, horizon):
            pred.iloc[len(timeseries) - center + i] = last
        pred = pred.iloc[:,0]

    timeseries = timeseries.iloc[:,0]
    wma = wma.iloc[:,0]

    if graph == True and horizon <= 0:
        style.use('ggplot')
        plt.plot(timeseries[:-limit])
        plt.plot(wma)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.legend()
        # plt.show()
        plt.savefig("wma.png")
    elif graph == True and horizon > 0:
        style.use('ggplot')
        plt.plot(timeseries[:-limit])
        plt.plot(wma)
        plt.plot(pred)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.legend()
        # plt.show()
        plt.savefig("wma.png")

    return wma, last
    ############### End of Function ##############

# Function: MA
def moving_average(timeseries, n = 2, graph = True, horizon = 0, limit=5):

    name = 'Moving Average (' + str(n) + ')'
    timeseries = pd.DataFrame(timeseries.values, index = timeseries.index, columns = [timeseries.name])/1.0
    ma = pd.DataFrame(np.nan, index = timeseries.index, columns = [name])
    center = int(n/2)
    start  = 0
    finish = n

    if(n % 2 == 0):
        weights = [None]*(n + 1)
        for i in range(0, n + 1):
            if (i == 0 or i == n):
                weights[i] = 1/(2*n)
            else:
                weights[i] = 1/n

        for i in range(center, len(timeseries) - center):
            ma.iloc[i, 0] = 0
            for j in range(0, len(weights)):
                ma.iloc[i, 0]  = float(ma.iloc[i, 0] + timeseries.iloc[start + j,:]*weights[j])

            start  = start  + 1

        last = ma.iloc[(start - 1),0]
    else:
        for i in range(center, len(timeseries) - center):
            ma.iloc[i,0]  = float(timeseries.iloc[start:finish,:].sum()/n)
            start  = start  + 1
            finish = finish + 1

        last = float(timeseries.iloc[(start-1):(finish-1),:].sum()/n)

    if horizon > 0:
        if len(timeseries) - center + horizon <= len(timeseries):
            time_horizon = len(timeseries)
        else:
            time_horizon = len(timeseries) - center + horizon
        time_horizon_index = pd.date_range(timeseries.index[0], periods = time_horizon, freq = timeseries.index.inferred_freq)
        pred = pd.DataFrame(np.nan, index = time_horizon_index, columns = ["Prediction"])
        for i in range(0, horizon):
            pred.iloc[len(timeseries) - center + i] = last
        pred = pred.iloc[:,0]

    timeseries = timeseries.iloc[:,0]
    ma = ma.iloc[:,0]

    if graph == True and horizon <= 0:
        style.use('ggplot')
        plt.plot(timeseries[:-limit])
        plt.plot(ma)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.legend()
        # plt.show()
        plt.savefig("ma.png")
    elif graph == True and horizon > 0:
        style.use('ggplot')
        plt.plot(timeseries[:-limit])
        plt.plot(ma)
        plt.plot(pred)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.legend()
        # plt.show()
        plt.savefig("ma.png")

    return ma, last
    ############### End of Function ##############

# Function: EMA
def exp_moving_average(timeseries, alpha = 0.5, graph = True, horizon = 0, optimize = False, limit =5):

    name = 'Exponential Moving Average'
    timeseries = pd.DataFrame(timeseries.values, index = timeseries.index, columns = [timeseries.name])/1.0
    ema = pd.DataFrame(np.nan, index = timeseries.index, columns = [name])
    ema.iloc[0,0]  = float(timeseries.iloc[0,:])

    if optimize == True:
        rms = float(timeseries.sum()**2)
        for var_alpha in range(0, 101):
            for i in range(1, len(timeseries)):
                ema.iloc[i,0]  = ema.iloc[i - 1, 0] + (var_alpha/100.0)*(float(timeseries.iloc[i - 1,:]) - ema.iloc[i - 1, 0])
                last = float(ema.iloc[i,0])
            print("Optimizing... Iteration ", var_alpha, " of 100")
            if rms >= sqrt(mean_squared_error(timeseries, ema)):
                rms = sqrt(mean_squared_error(timeseries, ema))
                opt_list = var_alpha/100.0, ema

        ema = opt_list[1]
        ema.columns = ['EMA(alpha=' + str( opt_list[0]) + ')']
        print("The optimal value for alpha = ", opt_list[0])
    else:
        for i in range(1, len(timeseries)):
            ema.iloc[i,0]  = ema.iloc[i - 1, 0] + alpha*(float(timeseries.iloc[i - 1,:]) - ema.iloc[i - 1, 0])
            last = float(ema.iloc[i,0])

    if horizon > 0:
        time_horizon = len(timeseries) + horizon
        time_horizon_index = pd.date_range(timeseries.index[0], periods = time_horizon, freq = timeseries.index.inferred_freq)
        pred = pd.DataFrame(np.nan, index = time_horizon_index, columns = ["Prediction"])
        for i in range(0, horizon):
            pred.iloc[len(timeseries) + i] = last
        pred = pred.iloc[:,0]

    rms = sqrt(mean_squared_error(timeseries, ema))
    timeseries = timeseries.iloc[:,0]
    ema = ema.iloc[:,0]

    if graph == True and horizon <= 0:
        style.use('ggplot')
        plt.plot(timeseries[:-limit])
        plt.plot(ema)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.legend()
        # plt.show()
        plt.savefig("ema.png")
    elif graph == True and horizon > 0:
        style.use('ggplot')
        plt.plot(timeseries[:-limit])
        plt.plot(ema)
        # plt.plot(pred)
        plt.title(timeseries.name)
        plt.ylabel('')
        plt.legend()
        # plt.show()
        plt.savefig("ema.png")

    return ema, last, rms
    ############### End of Function ##############

######################## Part 2 - Usage ####################################
# Load Dataset
# df = pd.read_csv('Python-Forecasting-01-Dataset.csv', sep = ',')
# df = pd.read_csv('../tweet_per_minute.csv', sep = ',')
# df = pd.read_csv('../facebook_active_users.csv', sep = ',')
df = pd.read_csv('youtube_hours.csv', sep = ',')

# Transforming the Dataset to a Time Series
X = df.iloc[:,:]
X = X.set_index(pd.DatetimeIndex(df.iloc[:,0]))
X = X.iloc[:,1]

# Calling Functions
# weighted_moving_average(X, w = [0.2, 0.3, 0.5], graph = True, horizon = 0, limit=5 )
# moving_average(X, n = 5, graph = True, horizon = 0, limit=5)
# exp_moving_average(X, alpha = 1, graph = True, horizon = 0, optimize = False, limit=5)


# weighted_moving_average(X, w = [0.2, 0.3, 0.5], graph = True, horizon = 0, limit=20 )
# moving_average(X, n = 5, graph = True, horizon = 0, limit=20)
# exp_moving_average(X, alpha = 1, graph = True, horizon = 0, optimize = False, limit=20)

# weighted_moving_average(X, w = [0.05,0.05, 0.5, 0.40], graph = True, horizon = 0, limit=6)
# moving_average(X, n = 5, graph = True, horizon = 0, limit=6)
exp_moving_average(X, alpha = 1, graph = True, horizon = 0, optimize = False, limit=7 )


########################## End of Code #####################################
