# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
import statsmodels
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import adfuller
# %% 
# data cleaning

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
    df_impute=dataframe[str(start):str(end)]
    df_impute=df_impute[df_impute.index.year!=imputeyear]

    #aggregate values for the sliced dataset
    agg1 = df_impute.groupby(df_impute.index.day_of_year).aggregate('mean')
    cursor=dataframe[dataframe[imputecolumn]==missingvalue].index
    fillslice = agg1.loc[cursor.day_of_year]['tg']

    # get the dates for the missing values
    dataframe.loc[cursor, imputecolumn]=list(fillslice)

    return(dataframe)

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

# add trend 
def get_trend(dataframe,y='tg'):
    ''' Adds a column containg the trend of the timeseries using linear regression
    Parameters
    ----------
    dataframe: pd.Dataframe: data input

    y: str: name of time series column

    Returns
    -------
    input data frame with trend column added
    
    '''
    # create timestep
    dataframe['timestep']=list(range(len(dataframe)))
    # fit lin_reg
    m=LinearRegression()
    m.fit(X=dataframe[['timestep']],y=dataframe[y])
    dataframe['trend']=pred=m.predict(dataframe[['timestep']])
    # dataframe=dataframe.drop('timestep',axis=1)
    print('Intercept:',m.intercept_,'\nSlope: ',m.coef_)
    return dataframe


# model seasonality
def get_seasonal(dataframe,y='tg',trend='trend'):
    ''' Using a linear regression model to predict seasonality of the time series
    ----------
    dataframe: pd.Dataframe: data input

    y: str: name of time seris column
    trend: str: name of trend column

    Returns
    -------
    data frame with trend column added
    
    '''
    seasonal_dummies = pd.get_dummies(dataframe.index.month,prefix='month')
    seasonal_dummies = seasonal_dummies.set_index(dataframe.index)
    dataframe=dataframe.join(seasonal_dummies)
    # dataframe.info()
    m=LinearRegression()
    X=dataframe.drop([y,trend],axis=1)
    #print(X)
    #print(dataframe[y])
    m.fit(X,dataframe[y])
    dataframe['trend_seasonal']=m.predict(X)
    # dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='month')))]
    dataframe['remainder'] = dataframe['tg'] - dataframe['trend_seasonal']
    return(dataframe)


# create lag columns: 
def lag(df,lagcount,rm_na=True):
    """Creates and visualizes time lagged-input feature(s) column in given data frame.

    Args:
        df (pd.DataFrame): Input data frame containing a remainder of a time series
        lagcount (int): Number of lag features 
        rm_na (bool, optional): Wether to remove NA values and calculate the autoregression. Defaults to True.

    Returns:
        _type_: DataFrame containing with lag columns added
    """
    
    fig, ax = plt.subplots(nrows=3, ncols=lagcount, figsize= ((lagcount*6),12)) 
    for _lag in range(lagcount):
        # Plotting lag vs remainder:
        colname = 'lag' + str(_lag+1)
        df.loc[:,colname] = df['remainder'].shift(int(_lag+1))
        if lagcount==1:
            sns.scatterplot(x= colname, y='remainder', data=df, ax=ax[0])
        else:
            sns.scatterplot(x= colname, y='remainder', data=df, ax=ax[0,_lag])

        # Drop missing values
        if rm_na is True:
            df.dropna(inplace=True)

        # Print correlation matrix
        if (_lag+1) == lagcount:
            corr_matrix = df[['tg'] + [col for col in df.columns if col.startswith('lag')]].corr()
            print(corr_matrix)

        # autoregression
        # Assign X and y
        if rm_na is True:
            X = df[[colname]]
            y = df['remainder']

            # Create and fit the model
            m = LinearRegression()
            m.fit(X, y)

            # Create predictions
            pred_col_name = 'pred_' + colname
            df[pred_col_name] = m.predict(X)

            # Plot the original remainder and the prediction
            if lagcount==1:
                sns.scatterplot(x= pred_col_name, y='remainder', data=df, ax=ax[1])
            else:    
                sns.scatterplot(x= pred_col_name, y='remainder', data=df, ax=ax[1,_lag])
        

            # # Is the remainder prediction error smaller than the remainder itself?
            df_plot = df.iloc[-365:].copy()
            _ylim = 180
            if lagcount==1:
                df_plot['remainder'].plot(ylim=[-_ylim,_ylim], legend =True,ax=ax[2])
                (df_plot['remainder'] - df_plot[pred_col_name]).plot(ylim=[-_ylim,_ylim],ax=ax[2])
                ax[2].legend(["Remainder", "Pred_Error"]);
            else:
                df_plot['remainder'].plot(ylim=[-_ylim,_ylim], legend =True,ax=ax[2,_lag])
                (df_plot['remainder'] - df_plot[pred_col_name]).plot(ylim=[-_ylim,_ylim],ax=ax[2,_lag],legend=True)
                ax[2,_lag].legend(["Remainder", "Pred_Error"],frameon=False);



    return df

# # add lag1, remove remainder
# def predict_full(df,df_val=df):
#     """Predict Time Series based using trend, seasonality and lag as input features.

#     Args:
#         df (pd.DataFrame): Training data frame containg the input features named 'Timestep', 'month_*' and 'lag*'. 
#         df2 (pd.DataFrame, optional):Validation data. Defaults to using training data instead of validation data.

#     Returns:
#         list: Predicted time series. Returns Predictions for Validation data if is given.
#     """
#     X_full = df.filter(regex='timestep|month_|^lag')
#     y_full = df['tg']
#     df_val_X = df_val.filter(regex='timestep|month_|^lag')
#     df_val_y = df_val['tg']
#     # fit linear regression
#     m_full = LinearRegression()
#     m_full.fit(X_full, y_full) 
#     print('r2:', m_full.score(df_val_X, df_val_y))
#     df2_X['pred'] = m_full.predict(df_val_X)    
#     return df_val_X['pred']



# %%
# Plot the data
def plot_remainder(df, 
                   title='Remainder after extracting trend and seasonality'):
    '''
    Plotting the remainder of a time series after removing trend and seasonality.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to plot.
    title : str
        The title of the plot
    
    Returns
    -------
    Plots the data
    '''
    
    df.plot()
    plt.title(title)
    plt.ylabel('Temperature [°C]')
    plt.show()


class XLocIndexer:
    '''
    Adding a custom pandas data frame indexing method, allowing 
    '''
    def __init__(self, frame):
        self.frame = frame
    
    def __getitem__(self, key):
        row, col = key
        return self.frame.iloc[row][col]

pd.core.indexing.IndexingMixin.xloc = property(lambda frame: XLocIndexer(frame))

# Usage
# df.xloc[0, 'a'] # one


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
