# %%
# Import packages
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

class TimeSeries:

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

    def autoregression(self, lagcount, rm_na=True):
        # intialize plots:
        fig, ax = plt.subplots(nrows=3, ncols=lagcount, figsize=((lagcount * 6), 12))
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
            if lagcount == 1:
                sns.scatterplot(x=lag_col_name, y='remainder', data=self.data, ax=ax[0])
            else:
                sns.scatterplot(x=lag_col_name, y='remainder', data=self.data, ax=ax[0, _lag])

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
                if lagcount == 1:
                    sns.scatterplot(x=pred_col_name, y='remainder', data=self.data, ax=ax[1])
                else:
                    sns.scatterplot(x=pred_col_name, y='remainder', data=self.data, ax=ax[1, _lag])
                
                # Third plot: remainder vs. prediction error
                # # Is the remainder prediction error smaller than the remainder itself?
                df_plot = self.data.iloc[-365:].copy()
                _ylim = 180
                if lagcount == 1:
                    df_plot['remainder'].plot(ylim=[-_ylim, _ylim], legend=True, ax=ax[2])
                    (df_plot['remainder'] - df_plot[pred_col_name]).plot(ylim=[-_ylim, _ylim], ax=ax[2])
                    ax[2].legend(["Remainder", "Pred_Error"])
                else:
                    df_plot['remainder'].plot(ylim=[-_ylim, _ylim], legend=True, ax=ax[2, _lag])
                    (df_plot['remainder'] - df_plot[pred_col_name]).plot(ylim=[-_ylim, _ylim], ax=ax[2, _lag], legend=True)
                    ax[2, _lag].legend(["Remainder", "Pred_Error"], frameon=False)

    def model_fit(self, x_cols="timestep|month_|^lag"):
        X = self.data.filter(regex=x_cols)
        self.model = LinearRegression()
        self.model.fit(X, self.data[self.y_column])
    
    def predict(self, x_cols="timestep|month_|^lag"):
        if self.model is None:
            print("Model not found. Fitting autoregressive model.")
            self.model_fit()

        X = self.data.filter(regex='timestep|month_|^lag')
        
        self.data['full_pred'] = self.model.predict(X)

        df_plot = self.data.iloc[-(2*365):].copy()
        ax = df_plot[[self.y_column,'full_pred']].plot()
        ax.set_title(f'Full model')
        plt.show()

df = pd.read_pickle('data/df_train')
df = df[['tg']]
ts = TimeSeries(df,'tg')

ts.autoregression(lagcount=1)
ts.predict()


# %%
