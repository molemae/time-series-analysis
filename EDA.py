# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
from datetime import date as dt

from utils import imputer, traintest, agg, get_trend, get_seasonal


# %%
# Read data
df_in = pd.read_csv('data/TG_STAID002759.txt',sep=',',header=14,names=('souid', 'date', 'tg','q_tg'))
df_in['date'] = pd.to_datetime(df_in['date'],format='%Y%m%d')
df_in=df_in.set_index(['date'])

# drop flag and id columns
df_in=df_in[['tg']] 
df_in.tail()

# %%
# impute missing data and pickle dataframe
df = imputer(df_in,1944,1947,1945,'tg',-9999)
df.to_pickle('data/df_full_imputed_pickle')

# train test split
df_train, df_test = traintest(df)

# aggregate data to monthly values
df_agg = agg(df_train)

# %% 
# add trend 

df_train =get_trend(df_train)
df_agg = get_trend(df_agg)
df_train.head()

# plot trend
df_plot=df_agg.reset_index().drop(columns='timestep')
df_plot = df_plot.melt('y-m',var_name='cols',value_name='value')
sns.lineplot(data=df_plot,x='y-m',y='value',hue='cols')
plt.xlim(dt(1980,1,1),dt(2020,12,31))

# model seasonality
df_agg = get_seasonal(df_agg)
df_agg.head()
df_train = get_seasonal(df_train)

# transform test data set
df_test = get_trend(df_test)
df_test = get_seasonal(df_test)
df_test.head()

# Pickle dataframe:
df_train.to_pickle(path='data/df_train')
df_test.to_pickle(path='data/df_test')

# %%
