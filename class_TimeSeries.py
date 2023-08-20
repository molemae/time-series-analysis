# %%
import pandas as pd
from sklearn.linear_model import LinearRegression

class TimeSeriesFrame:

    def __init__(self, data, y_column):
        self.data = data.copy()
        self.y_column = y_column
        self.model = None
        self.plot_acf = None
        self.plot_pacf = None
        self.trend()
        self.seasonality()
        self.remainder()
        
    def trend(self):
        self.data['timestep'] = list(range(len(self.data)))
        X = self.data[['timestep']]
        m = LinearRegression()
        m.fit(X, self.data[self.y_column])
        self.data['trend'] = m.predict(X)
        print('Trend: Linear Regressio', '\nIntercept:', m.intercept_, '\nSlope: ', m.coef_)

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
    
    def autoregression(df,lagcount,rm_na=True):
        """Creates and visualizes time lagged-input feature(s) of the remainder in given data frame.

        Args:
            df (pd.DataFrame): Input data frame containing a remainder of a time series
            lagcount (int): Number of lag features 
            rm_na (bool, optional): Wether to remove NA values and calculate the autoregression. Defaults to True.

        Returns:
            _type_: DataFrame containing with lag columns added
        """
        
        # Plotting lag vs remainder:
        fig, ax = plt.subplots(nrows=3, ncols=lagcount, figsize= ((lagcount*6),12)) 
        for _lag in range(lagcount):
            # create column
            lag_col_name = 'lag' + str(_lag+1)
            df.loc[:,lag_col_name] = df['remainder'].shift(int(_lag+1))
            
            # Drop missing values
            if rm_na is True:
                df.dropna(inplace=True)

            # Print correlation matrix
            if (_lag+1) == lagcount:
                corr_matrix = df[['tg'] + [col for col in df.columns if col.startswith('lag')]].corr()
                print('Correlatoin matrix: lag vs remainder:\n', corr_matrix)

            # First plot: lag vs remainder:
            if lagcount==1:
                sns.scatterplot(x= lag_col_name, y='remainder', data=df, ax=ax[0])
            else:
                sns.scatterplot(x= lag_col_name, y='remainder', data=df, ax=ax[0,_lag])

            # auto regressive model:
            if rm_na is True:
                # Select the model's X and y
                X = df[[lag_col_name]]
                y = df['remainder']

                # Call and fit the model
                m = LinearRegression()
                m.fit(X, y)

                # Create predictions
                pred_col_name = 'pred_' + lag_col_name
                df[pred_col_name] = m.predict(X)

                # Second plot: the remainder vs prediction
                if lagcount==1:
                    sns.scatterplot(x= pred_col_name, y='remainder', data=df, ax=ax[1])
                else:    
                    sns.scatterplot(x= pred_col_name, y='remainder', data=df, ax=ax[1,_lag])
            
                # Third plot:
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

    # def ts_autoregression(self, lagcount=1, rm_na=True):
    #     for _lag in range(lagcount):
    #         colname = 'lag' + str(_lag+1)
    #         self.data[colname] = self.data['remainder'].shift(int(_lag+1))
    #         if rm_na is True:
    #             self.data.dropna(inplace=True)
    #     ''' add lag() functionality:
    #         - lag() function predicts and adds the column only for the lag column
    #         - add prediction for full model!
    #         '''

    def ts_model_fit(self, x_cols="timestep|month_|^lag"):
        X = self.data.filter(regex=x_cols)
        self.model = LinearRegression()
        self.model.fit(X, self.data[self.y_column])

    def ts_predict(self, X=None, y=None):
        if self.model is None:
            print("Model not found. Fitting autoregressive model.")
            self.ts_model_fit()
        
        if X is None:
            X = self.data.filter(regex='timestep|month_|^lag')
        
        if y is None:
            y = self.y_column
        
        return self.model.predict(X)

df = pd.read_pickle('data/df_train')
df = df[['tg']]
ts = TimeSeriesFrame(df,'tg')

ts.data.head()

# %%
