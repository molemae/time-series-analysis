# %%
# Importing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import ar_select_order

# import functions from local file:
from utils import lag, plot_remainder,XLocIndexer
# from time_series_analysis import get_trend,get_seasonal


# Set figure size to (14,6)
plt.rcParams['figure.figsize'] = (14,6)
plt.rcParams['font.size'] = '14'

# pandas stuff
pd.options.mode.chained_assignment = None


# %%
# read data transformd in 'time_series_analysis.py':
df = pd.read_pickle('data/df_train')
df_test = pd.read_pickle('data/df_test')
# plot_remainder(df)
df.head()
# %% 
# create lag: 
df = lag(df,1)

# %%
# Plot ACF and PACF
fig,ax = plt.subplots(nrows=1, ncols=2, figsize= (12,6))
plot_acf(df['remainder'],ax=ax[0])
plt.xlabel('lags')
plot_pacf(df['remainder'],ax=ax[1]);12

# %%
#  Box-Jenkins-Methodology
lags_order = ar_select_order(df['remainder'], maxlag=5)

# %% 
# --> Full model
# Creating Full Model

# add lag1, remove remainder
def predict_full(df,df_val=df):
    """Predict Time Series based using trend, seasonality and lag as input features.

    Args:
        df (pd.DataFrame): Training data frame containg the input features named 'Timestep', 'month_*' and 'lag*'. 
        df2 (pd.DataFrame, optional):Validation data. Defaults to using training data instead of validation data.

    Returns:
        list: Predicted time series. Returns Predictions for Validation data if is given.
    """
    # create
    X_full = df.filter(regex='timestep|month_|^lag')
    y_full = df['tg']
    df_val_X = df_val.filter(regex='timestep|month_|^lag')
    df_val_y = df_val['tg']
    # fit linear regression
    m_full = LinearRegression()
    m_full.fit(X_full, y_full) 
    print('r2:', m_full.score(df_val_X, df_val_y))
    df['pred'] = m_full.predict(df_val_X)    
    return df_val_X['pred']



df['pred_full'] = predict_full(df)
    #plot
df.loc['1980':'2020',['tg','pred_full']].plot()

# %%
# CrossValidation:
def cross_val(df,n_splits=5):
    """ Calculates cross validation scores for times series.

    Args:
        df (_type_): _description_
    """
    X_full = df.filter(regex='timestep|month_|^lag')
    y_full = df['tg']
    m_full = LinearRegression()
    m_full.fit(X_full, y_full)

    # Create a TimeSeriesSplit object
    ts_split = TimeSeriesSplit(n_splits=n_splits)

    # Create the time series split
    time_series_split = ts_split.split(X_full, y_full)

    # CrossVal
    crossval = cross_val_score(estimator=m_full, X=X_full, y=y_full, cv=time_series_split)
    print(crossval, '\nMean: ', round(crossval.mean(), 3))

cross_val(df)


# %% 
# Prediction

# ad lag to and remove columns from test data set 
df_test = lag(df_test,1,rm_na = False)
# add 
df_test.loc['2022-01-01','lag1'] = df.loc['2021-12-31','tg']
df_test.head()

# %%
# Cobine data train and test frames
df_all = pd.concat([df.drop('pred_full',axis=1),df_test],).drop(['trend','trend_seasonal','remainder','pred_lag1'],axis=1)

# Fit lm to full data frame
X_all = df_all.drop(['tg'],axis=1)
y_all = df_all['tg']

m_all = LinearRegression()
m_all.fit(X_all, y_all)
m_all.predict(X_all)

# %%
# create future data step
timestep = [df_all['timestep'].max()+1]
months = [1]+[0]*11
lag1 = df_all.xloc[-1,'tg']

X_future =timestep
X_future.extend(months)
X_future.append(lag1)

X_future = pd.DataFrame([X_future])
X_future.columns = X_all.columns

# %%
# predict

m_all.predict(X_future)


# %%
