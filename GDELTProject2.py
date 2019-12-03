#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.tsa
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tools.eval_measures import rmse, aic


#read in csv
#new_dataTS.csv is modified to align fatalities with the same month as the GDELT data and is all training data
master=pd.read_csv("new_dataTS.csv")
master['YearMonth']=pd.to_datetime(master['YearMonth'])



#Make dataframe of training data for SSD
df=master[master['Country_Code'] == 'SSD']
#

#Drop Country code column and index by date
df=df.drop('Country_Code', axis=1)
df=df.rename(columns={"YearMonth": "date"})
df=df.set_index('date')


#following VAR example on https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

# Plot
# fig, axes = plt.subplots(nrows=8, ncols=5, dpi=120, figsize=(5,3))
# for i, ax in enumerate(axes.flatten()):
#     data = df[df.columns[i]]
#     ax.plot(data, color='red', linewidth=1)
#     # Decorations
#     ax.set_title(df.columns[i])
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#     ax.spines["top"].set_alpha(0)
#     ax.tick_params(labelsize=6)
#
# plt.tight_layout();


#Granger Causality
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(df, variables = df.columns)

#Cointegration test


def cointegration_test(df, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df)


#Make train and test sets
df_train= df

#Augmented Dickey Fuller test
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")

# ADF Test on each column
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# 1st difference
df_differenced = df_train.diff().dropna()

for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# Second Differencing
df_differenced = df_differenced.diff().dropna()

for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')
#All variables are stationary

model = VAR(df_differenced)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

#Lag order 2 has lowest AIC
model_fitted = model.fit(2)
model_fitted.summary()


from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print(col, ':', round(val, 2))


# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)

# # Input data for forecasting
# forecast_input = df_train.values[-lag_order:]
# forecast_input
forecast_input = df_differenced.values[-lag_order:]
forecast_input

# Forecast
#fc = model_fitted.forecast(y=forecast_input, steps=nobs)
fc = model_fitted.forecast(forecast_input, 1)
df_forecast = pd.DataFrame(fc, index=df.index[-1:], columns=df.columns + '_2d')

df_forecast

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(df_train, df_forecast, second_diff=True)


# fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
# for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
#     df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
#     df_test[col][-1:].plot(legend=True, ax=ax);
#     ax.set_title(col + ": Forecast vs Actuals")
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#     ax.spines["top"].set_alpha(0)
#     ax.tick_params(labelsize=6)
#
# plt.tight_layout();





#Try VARMAX

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

#exog = endog['dln_consump']
exog = df_train['fatalities']
endog = df_train.drop('fatalities', axis=1)
mod = sm.tsa.VARMAX(endog, order=(2,0), trend='nc', exog=exog)
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())

mod = sm.tsa.VARMAX(df_differenced, order=(2,0), trend='nc')
mod=sm.tsa.VARMAX(df_train)
res = mod.fit(maxiter=1000, disp=True)
print(res.summary())

res.forecast(steps=1)

mod=sm.tsa.VARMAX(df_train[['G-P Nodes', 'G-I Nodes', 'B-N Nodes', 'N-P Nodes', 'fatalities']], order=(2,0), trend='nc')
mod=sm.tsa.VARMAX(df_train[['G-P Nodes', 'G-I Nodes', 'B-N Nodes', 'N-P Nodes', 'fatalities']], trend='nc')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())

res.forecast(steps=1)
fcast_res1 = res.get_forecast(steps=1)

# Most results are collected in the `summary_frame` attribute.
print(fcast_res1.summary_frame(4))



import statsmodels.api as sm
X=df.iloc[:, 1:38]
y=df.iloc[:, 38]
model=sm.OLS(y,X).fit()
predictions=model.predict(X)
model.summary()
predictions

pd.concat([y,predictions], axis=1)

#Ordinal Brier score for 5 bin questions

#forecast
f=[0.2, 0.2, 0.2, 0.2, 0.2]
#actual
a=[1,0,0,0,0]

def ord_brier(forecast, actual):
    #find squared error of each cumulative group of bins
    bin1cumulativeerror=(sum(forecast[:1])-sum(actual[:1]))**2
    bin2cumulativeerror=(sum(forecast[:2])-sum(actual[:2]))**2
    bin3cumulativeerror=(sum(forecast[:3])-sum(actual[:3]))**2
    bin4cumulativeerror=(sum(forecast[:4])-sum(actual[:4]))**2
    #bin5cumalativeerror will always be 0
    totalerror=bin1cumulativeerror+bin2cumulativeerror+bin3cumulativeerror+bin4cumulativeerror
    score=totalerror/4
    return score

#Brier score of random guess...
ord_brier(f,a)
#exact match should produce a 0 Brier score...
ord_brier([1,0,0,0,0],[1,0,0,0,0])
#worst forecast should produce a 1 Brier score...
ord_brier([0,0,0,0,1],[1,0,0,0,0])
#random guess...
ord_brier([0.2,0.2,0.2,0.2,0.2],[0,0,0,1,0])



#Univariate ARIMA on fatalities
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
import matplotlib.pyplot as plt
y.plot()
#df.plot()
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(y)
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(y, order=(2,1,0))
model_fit=model.fit(disp=0)
print(model_fit.summary())

residuals=pd.DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind='kde')
print(residuals.describe())

model_fit.forecast()

import math
math.log(10)
math.exp(math.log(10))

y2=math.log(y)
y2=[]
for i in range(len(y)):
    #y2[i]=float(math.log(y.iloc[i]))
    #print(y[i])
    y2.append(math.log(y.iloc[i]))

model2=ARIMA(y2, order=(2,1,0))
model_fit2=model2.fit(disp=0)
print(model_fit2.summary())

residuals2=pd.DataFrame(model_fit2.resid)
residuals2.plot()
residuals2.plot(kind='kde')
print(residuals2.describe())

model_fit2.forecast()

math.exp(model_fit2.forecast()[0][0])

math.exp(model_fit2.forecast()[2][0][0])
math.exp(model_fit2.forecast()[2][0][1])



import statistics
statistics.stdev(y)

#VAR on multiple country fatalities



#ARIMA plus regression one country at a time
#start with shifted response (use new_data.csv)
master2=pd.read_csv("new_data.csv")
master2['Year/Month']=pd.to_datetime(master2['Year/Month'])



#Make dataframe of training data for SSD
df2=master2[master2['Country_Code'] == 'SSD']
#

#Drop Country code column and index by date

df2=df2.drop('Country_Code', axis=1)
df2=df2.drop('Unnamed: 0', axis=1)
df2=df2.drop('month', axis=1)
df2=df2.rename(columns={"Year/Month": "date"})
df2=df2.set_index('date')

df2net=df2.loc[:,['G-P Nodes', 'G-I Nodes', 'B-N Nodes', 'N-P Nodes', 'fatalities']]
#df2event=df2.loc[:,['1', '9', '10', '15', 'fatalities']]
df2event=df2.loc[:,['1', '9', '10', '15', 'fatalities']]
#Take log of fatalities
#df2net['fatalities']=np.log(df2net['fatalities'])

#Make a regression model for each three months at a time
#make table for coeficients:
coeftab=[]
for i in range(len(df2net)-4): #don't use last row
    #build linear model
    rmod = sm.OLS(df2net['fatalities'].iloc[i:i+3], df2net.drop('fatalities', axis=1).iloc[i:i+3]).fit()
    coeftab.append(rmod.params)

#make time series of coefficients
GP=[]
GI=[]
BN=[]
NP=[]
for i in range(len(coeftab)):
    GP.append(coeftab[i][0])
    GI.append(coeftab[i][1])
    BN.append(coeftab[i][2])
    NP.append(coeftab[i][3])

#Predict next coefficient values...
nextcoef=[]

model3 = ARIMA(GP, order=(2, 1, 0))
model_fit3 = model3.fit(disp=0)
nextcoef.append(model_fit3.forecast()[0][0])
model3 = ARIMA(GI, order=(2, 1, 0))
model_fit3 = model3.fit(disp=0)
nextcoef.append(model_fit3.forecast()[0][0])
model3 = ARIMA(BN, order=(2, 1, 0))
model_fit3 = model3.fit(disp=0)
nextcoef.append(model_fit3.forecast()[0][0])
model3 = ARIMA(NP, order=(2, 1, 0))
model_fit3 = model3.fit(disp=0)
nextcoef.append(model_fit3.forecast()[0][0])


#July numbers...
nextfeatures=df2net.drop('fatalities', axis=1).iloc[len(df2net)-1]
#Prediction
sum(nextcoef*nextfeatures)
#if log
#np.exp(sum(nextcoef*nextfeatures))
#actual
df2net['fatalities'].iloc[len(df2net)-1]

from scipy.stats import norm

#Declare cutpoints:
#For COD:
cuts=[51, 69, 91, 129]
#For CAF
cuts=[10, 30, 50, 100]
#For SSD
cuts=[80, 130, 210, 310]
#For ETH
cuts=[33, 50, 70, 105]

#Declare answers (for SSD, COD, ETH):
a=[1,0,0,0,0]
#For CAF (351):
a=[0,0,1,0,0]

pdist = norm(loc=sum(nextcoef*nextfeatures), scale = statistics.stdev(df2net['fatalities'].iloc[0:(len(df2net)-1)]))
bin1 = pdist.cdf(cuts[0])
bin2 = pdist.cdf(cuts[1]) - bin1
bin3 = pdist.cdf(cuts[2]) - (bin1+bin2)
bin4 = pdist.cdf(cuts[3]) - (bin1+bin2+bin3)
bin5 = 1 - (bin1+bin2+bin3+bin4)

#Brier score
f=[bin1,bin2, bin3, bin4, bin5]

ord_brier(f,a)




#Calculate percent humans beat
#read-in human forecasts
humans=pd.read_csv("RCTAforecasts.csv")
humans['Date']=pd.to_datetime(humans['Date'])

recent=humans.sort_values('Date', ascending=True)
recent=recent.drop_duplicates(['discover_id','user_id'], keep='first')

Brier=[]
for i in range(len(recent)):
    f=[recent.iloc[i,5],recent.iloc[i,6],recent.iloc[i,7],recent.iloc[i,8],recent.iloc[i,9]]
    if recent.iloc[i,0] == 351:
        a = [0, 0, 1, 0, 0]
    else:
        a = [1, 0, 0, 0, 0]
    Brier.append(ord_brier(f,a))

recent['Brier']=Brier

def percentbetter(IFP,brier):
    #count number of forecasts for IFP
    df=recent[recent.discover_id==IFP]
    df2=df[df.Brier>brier]
    return(len(df2)/len(df))


#Calculate percent better for the ARIMARegression model
percentbetter(371,.231)
percentbetter(351,.153)
percentbetter(378,0)
percentbetter(311,0)

#For RF:
RF=[]
RF.append(percentbetter(351,.1469))
RF.append(percentbetter(311,.3077))
RF.append(percentbetter(378,.5213))
RF.append(percentbetter(371,.4549))
sum(RF)/4

ADA=[]
ADA.append(percentbetter(351,.1415))
ADA.append(percentbetter(311,.2443))
ADA.append(percentbetter(378,.2529))
ADA.append(percentbetter(371,.3977))
sum(ADA)/4

XGB=[]
XGB.append(percentbetter(351,.1311))
XGB.append(percentbetter(311,.2385))
XGB.append(percentbetter(378,.2881))
XGB.append(percentbetter(371,.4371))
sum(XGB)/4

Tree=[]
Tree.append(percentbetter(351,.0942))
Tree.append(percentbetter(311,.1034))
Tree.append(percentbetter(378,.5666))
Tree.append(percentbetter(371,.5485))
sum(Tree)/4

Ridge=[]
Ridge.append(percentbetter(351,.1340))
Ridge.append(percentbetter(311,.1110))
Ridge.append(percentbetter(378,.0518))
Ridge.append(percentbetter(371,.1342))
sum(Ridge)/4

Lasso=[]
Lasso.append(percentbetter(351,.1266))
Lasso.append(percentbetter(311,.2394))
Lasso.append(percentbetter(378,.2606))
Lasso.append(percentbetter(371,.3119))
sum(Lasso)/4

SVR=[]
SVR.append(percentbetter(351,.1707))
SVR.append(percentbetter(311,.2774))
SVR.append(percentbetter(378,.4130))
SVR.append(percentbetter(371,.0923))
sum(SVR)/4

NB=[]
NB.append(percentbetter(351,.1153))
NB.append(percentbetter(311,.1587))
NB.append(percentbetter(378,.1491))
NB.append(percentbetter(371,.1338))
sum(NB)/4

AR=[]
AR.append(percentbetter(351,.498))
AR.append(percentbetter(311,.498))
AR.append(percentbetter(378,.498))
AR.append(percentbetter(371,.498))
sum(AR)/4

vAR=[]
vAR.append(percentbetter(351,.588))
vAR.append(percentbetter(311,.588))
vAR.append(percentbetter(378,.588))
vAR.append(percentbetter(371,.588))
sum(vAR)/4

ord_brier([0,0,1,0,0],a)

lastmonth=[]
lastmonth.append(percentbetter(351,.25))
lastmonth.append(percentbetter(311,.75))
lastmonth.append(percentbetter(378,.75))
lastmonth.append(percentbetter(371,.5))
sum(lastmonth)/4

avgmonth=[]
avgmonth.append(percentbetter(351,.25))
avgmonth.append(percentbetter(311,.75))
avgmonth.append(percentbetter(378,.75))
avgmonth.append(percentbetter(371,.5))
sum(avgmonth)/4

random=[]
random.append(percentbetter(351,.09))
random.append(percentbetter(311,.3))
random.append(percentbetter(378,.3))
random.append(percentbetter(371,.3))
sum(random)/4

ord_brier([.05, .61, .24, .05, .05], [1, 0, 0, 0, 0])