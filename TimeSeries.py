# %%
# Import packages
import pandas as pd
import numpy as np
import re

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

class TimeSeries:

    def __init__(self, data, y_column):
        self.data = data.copy()
        self.y_column = y_column
        self.x_columns = "timestep|month_|^lag"
        self.lagcount = None
        self.X_full = None
        self.X_forecast = None
        self.model = None
        self.plot_acf = None
        self.plot_pacf = None
        self.predict_called = False # Flag 
        self.trend()
        self.seasonality()
        self.remainder()
        
    def trend(self,print_plot=False):
        self.data['timestep'] = list(range(len(self.data)))
        X = self.data[['timestep']]
        m = LinearRegression()
        m.fit(X, self.data[self.y_column])
        self.data['trend'] = m.predict(X)
        print('Trend: Linear Regression', '\nIntercept:', m.intercept_, '\nSlope: ', m.coef_)
        if print_plot:
            self.data.plot()


    def seasonality(self):
        seasonal_dummies = pd.get_dummies(self.data.index.month, prefix='month')
        seasonal_dummies = seasonal_dummies.set_index(self.data.index)
        self.data = pd.concat([self.data, seasonal_dummies], axis=1)
        X = self.data.filter(regex='month', axis=1)
        m = LinearRegression()
        m.fit(X, self.data[self.y_column])
        self.data['seasonal'] = m.predict(X)

    def remainder(self):
        X = self.data.filter(regex='timestep|month', axis=1)
        m = LinearRegression()
        m.fit(X, self.data[self.y_column])
        self.data['trend_seasonal'] = m.predict(X)
        self.data['remainder'] = self.data[self.y_column] - self.data['trend_seasonal']

    def autoregression(self, lagcount, rm_na=True, print_plot=True):
        if print_plot:
            # plot pacf and acf
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            plot_acf(self.data['remainder'], ax=ax[0])
            plt.xlabel('lags')
            plot_pacf(self.data['remainder'], ax=ax[1])

            # intialize lag plots:
            fig, ax = plt.subplots(nrows=3, ncols=lagcount, figsize=((lagcount * 6), 12))
            plt.subplots_adjust(hspace=0.5)
        
        for _lag in range(lagcount):
            # create column
            lag_col_name = 'lag' + str(_lag + 1)
            self.data.loc[:, lag_col_name] = self.data['remainder'].shift(int(_lag + 1))

            # Drop missing values
            if rm_na is True:
                self.data.dropna(inplace=True)
            
            # Print correlation matrix
            if (_lag + 1) == lagcount:
                corr_matrix = self.data[[self.y_column] + [col for col in self.data.columns if col.startswith('lag')]].corr()
                print('Correlation matrix: lag vs remainder:\n', corr_matrix)

            # First plot: lag vs remainder
            if print_plot:
                if lagcount == 1:
                    sns.scatterplot(x=lag_col_name, y='remainder', data=self.data, ax=ax[0])
                    ax[0].set_title(f'{lag_col_name} vs Remainder')
                else:
                    sns.scatterplot(x=lag_col_name, y='remainder', data=self.data, ax=ax[0, _lag])
                    ax[0, _lag].set_title(f'{lag_col_name} vs Remainder')

            # auto regressive modelling
            if rm_na is True:
                # Call and fit auto regressive model
                X = self.data[[lag_col_name]]
                y = self.data['remainder']
                ar_model = LinearRegression()
                ar_model.fit(X, y)

                # predict remainder
                pred_col_name = 'pred_' + lag_col_name
                self.data[pred_col_name] = ar_model.predict(X)

                # Second plot: the remainder vs prediction
                if print_plot:
                    if lagcount == 1:
                        sns.scatterplot(x=pred_col_name, y='remainder', data=self.data, ax=ax[1])
                        ax[1].set_title(f'{pred_col_name} vs Remainder')
                    else:
                        sns.scatterplot(x=pred_col_name, y='remainder', data=self.data, ax=ax[1, _lag])
                        ax[1, _lag].set_title(f'{pred_col_name} vs Remainder')
                    
                    # Third plot: remainder vs. prediction error
                    # # Is the remainder prediction error smaller than the remainder itself?
                    df_plot = self.data.iloc[-365:].copy()
                    _ylim = 180
                    if lagcount == 1:
                        df_plot['remainder'].plot(ylim=[-_ylim, _ylim], legend=True, ax=ax[2])
                        (df_plot['remainder'] - df_plot[pred_col_name]).plot(ylim=[-_ylim, _ylim], ax=ax[2])
                        ax[2].set_title('Remainder and Prediction Error')
                        ax[2].legend(["Remainder", "Pred_Error"])
                    else:
                        df_plot['remainder'].plot(ylim=[-_ylim, _ylim], legend=True, ax=ax[2, _lag])
                        (df_plot['remainder'] - df_plot[pred_col_name]).plot(ylim=[-_ylim, _ylim], ax=ax[2, _lag], legend=True)
                        ax[2, _lag].set_title('Remainder and Prediction Error')
                        ax[2, _lag].legend(["Remainder", "Pred_Error"], frameon=False)
        # save number of lags:
        self.lagcount = lagcount

    def model_fit(self):
        # check if lag was created
        if not any(re.search('^lag', s) for s in list(self.data.columns)):
            print('Warning: no time lagged feature was found. Call autoregresson method to create lag feature(s)')
            return None
            
        self.X_full = self.data.filter(regex=self.x_columns)
        self.model = LinearRegression()
        self.model.fit(self.X_full, self.data[self.y_column])
        
    def predict(self, print_plot=True):
        if self.model is None:
            print("Warning: Model not found. Fitting autoregressive model.")
            self.model_fit()
        
        self.data['full_pred'] = self.model.predict(self.X_full)

        if print_plot:
            df_plot = self.data.iloc[-(2*365):].copy()
            ax = df_plot[[self.y_column,'full_pred']].plot()
            ax.set_title(f'Full model')
            plt.show()
        self.predict_called = True
        
    # CrossValidation:
    def cross_val(self,n_splits=5):
        """ Calculates cross validation scores for times series."""
        if not self.predict_called:
            print("Warning: No predictions found. Running predictions.")
            self.predict(print_plot=False)

        # Create a TimeSeriesSplit object
        ts_split = TimeSeriesSplit(n_splits=n_splits)

        # Create the time series split
        time_series_split = ts_split.split(self.X_full, self.data[self.y_column])

        # CrossVal
        crossval = cross_val_score(estimator=self.model,
                                   X=self.X_full,
                                   y=self.data[self.y_column],
                                   cv=time_series_split)
        
        print('Cross Validation Scores:\n', crossval, '\nMean: ', round(crossval.mean(), 3))
    
    # create future data step
    def forecast(self):
        # get index
        index = self.data.index[-1]

        # create timestep
        timestep = [self.data.loc[index,'timestep']+1]

        # generate month columns 
        month_columns = np.zeros((1,12),dtype=int)
        forecast_month = index.month + 1
        if (forecast_month) > 12:
            forecast_month -= 13
        
        month_columns[forecast_month] = 1

        # add lag value (last y value in data frame)
        index_lag = self.data.index[-self.lagcount:]
        lag1 = self.data.loc[index_lag,self.y_column]

        self.X_forecast = pd.concat(
            (pd.DataFrame(np.array(timestep)),
            pd.DataFrame(month_columns),
            pd.DataFrame(np.array(lag1).reshape(1,self.lagcount))),
            axis=1
            )

        # forecast
        y_forecast = self.model.predict(self.X_forecast)
        
        # combine last data points with last week 
        df_plot = self.data.copy().iloc[-7:,[0]]
        df_forecast = pd.DataFrame(
            {self.y_column: y_forecast},
            index = [index+pd.DateOffset(months=1)])
        
        # plot forecast and last week of time series
        df_plot = pd.concat([df_plot,df_forecast])
        ax = sns.lineplot(data=df_plot, marker='o', markersize=7,)
        ax.plot(df_forecast.index, df_forecast['tg'], marker='o', markersize=9)
        plt.title('TimeSeries: Forecast')
        plt.ylabel('Temperature [0.1°C]')
        plt.show()
        
        return None

# %%
