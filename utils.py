# %%
import pandas as pd
import numpy as np
import plotly.express as px

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import adfuller
# %% 
# data cleaning

# decribe the time series, basic EDA:
def eda_describe(dataframe,y_column, missing_value=None):
    """ Basic EDA of given time series

    Args:
        dataframe (pd.DataFrame): DataFrame containing the time series
        y_column (str): name of time column
        missing_value (, optional): value of missing values. Defaults to None.
    """    
    # plot original time series:
    dataframe.plot()

    # print min, max and gap:
    print('Min:', dataframe[y_column].min(),
        '\nMax:', dataframe[y_column].max(),
        f"""\nGap: {dataframe[dataframe[y_column]==missing_value].index.min()} 
        - {dataframe[dataframe[y_column]==missing_value].index.max()}"""
        )

def time_series_plot(df, y_column):
    """
    Create a scrollable time series plot with vertical zooming using Plotly.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a date column and y_column.
        y_column (str): The column to visualize on the y-axis.
    """
    fig = px.scatter(df, y=y_column)
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(rangeslider_visible=False, rangeselector=dict(buttons=list([
        dict(count=7, label='1w', step='day', stepmode='backward'),
        dict(count=1, label='1m', step='month', stepmode='backward'),
        dict(count=3, label='3m', step='month', stepmode='backward'),
        dict(count=6, label='6m', step='month', stepmode='backward'),
        dict(count=1, label='1y', step='year', stepmode='backward'),
        dict(step='all')
    ])))
    fig.show()


# impute missing data 
def imputer(dataframe,start,end,imputeyear,imputecolumn,missingvalue):
    ''' Imputes missing daily data from existing data in years before and after the gap.
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        timeseries containing a gap

    start:int
        first year of slice to impute from 
    
    end: int
        last year of slice to impute from

    imputeyear: int
        year containing missing values

    imputecolumn: str
        name column where to impute
    
    missingvalue: int
        value indicating missing values

    Returns: pd.DataFrame
    -------
    dataframe containing imputed column 

    '''
    #create a data slice to impute from (start to end without the year containing the gap)
    df_copy = dataframe.copy()
    df_impute = df_copy[str(start):str(end)]
    df_impute=df_impute[df_impute.index.year!=imputeyear]

    #aggregate values for the sliced dataset
    agg1 = df_impute.groupby(df_impute.index.day_of_year).aggregate('mean')
    cursor=dataframe[dataframe[imputecolumn]==missingvalue].index
    fillslice = agg1.loc[cursor.day_of_year]['tg']

    # get the dates for the missing values
    df_copy.loc[cursor, imputecolumn]=list(fillslice)

    return df_copy

# train test split
def traintest(dataframe,split=365):
    ''' Splits the time series into train and test data. Test data is taken from last value to split value.
    
    Parameters
    ----------
    dataframe: pd.DataFrame: times series dataframe

    split: int: length of test data frame.  

    Returns
    -------
    train and test data frames

    '''
    if split <= 1:
        split = round(len(dataframe)*split)
    else:
        split= -1*split
    df_train = dataframe.copy().iloc[:split]
    df_test = dataframe.copy().iloc[split:]
    return df_train,df_test

# aggregate data 
def agg(dataframe,aggmethod='mean'):
    ''' Aggregates temperature values to monthly values
    Parameters
    ----------
    dataframe: pd.Dataframe: data input

    aggmethod: str: pandas groupby aggregation method

    Returns
    -------
    Aggregated dataframe
    
    '''
    # aggregate by year and month
    dataframe =dataframe.groupby([dataframe.index.year,dataframe.index.month]).aggregate(aggmethod)
    # reset index
    dataframe=dataframe.reset_index(names=['year','month'])
    dataframe['y-m']=dataframe['year'].astype('str')+'-'+dataframe['month'].astype('str')
    dataframe=dataframe.drop(['year','month'],axis=1)
    dataframe['y-m']=pd.to_datetime(dataframe['y-m'].astype('str'),format=('%Y-%m'))
    dataframe=dataframe.set_index('y-m')

    return dataframe


def print_adf(data):
    
    """ Prints the results of the Augmented Dickey Fuller Test.
    
    Parameters
    ----------
    data: 
        data for ad fuller test
    
    Returns
    -------
    None.
    """
    
    adf_stats, p, used_lag, n_obs, levels, information_criterion = adfuller(data)
    
    print(f"""adf_stats: {adf_stats}
            p: {p} 
            used lag: {used_lag} 
            number of observations: {n_obs}
            CI 99%: {levels['1%']}
            CI 95%: {levels['5%']}
            CI 90%: {levels['10%']}
            information criterion (AIC): {information_criterion}
            """)
# %%
