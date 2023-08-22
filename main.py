# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
from datetime import date as dt
from TimeSeries import TimeSeries
from utils import eda_describe,time_series_plot, imputer, agg, traintest

# %%
# Read data
df_in = pd.read_csv('data/TG_STAID002759.txt',sep=',',header=14,names=('souid', 'date', 'tg','q_tg'))
df_in['date'] = pd.to_datetime(df_in['date'],format='%Y%m%d')
df_in=df_in.set_index(['date'])

# drop flag and id columns
df_in=df_in[['tg']] 
df_in.tail()

# %%
# EDA:
eda_describe(df_in,'tg',-9999)
time_series_plot(df_in,'tg')
# %%
# impute missing data and pickle dataframe
df = imputer(df_in,1944,1947,1945,'tg',-9999)
df.to_pickle('data/df_full_imputed_pickle')

# aggregate data to monthly values
df = agg(df)

# train test split
df_train, df_test = traintest(df)


######################################

# %%
# Splitting Time Series and predicting
ts = TimeSeries(df_train,'tg')

ts.autoregression(lagcount=1)
ts.predict()
ts.cross_val()

print('\nValidation data\n')

df_test=pd.read_pickle('data/df_test')
df_test = df[['tg']]
ts_test = TimeSeries(df,'tg')

ts_test.autoregression(lagcount=1,print_plot=False)
ts_test.predict(print_plot=False)
ts_test.cross_val()

# %%
# Forecast
ts_full = TimeSeries(df,'tg')
ts_full.autoregression(2)
ts_full.predict()
ts_full.forecast()

# %%
