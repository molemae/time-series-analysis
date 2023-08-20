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
        self.ts_trend()
        self.ts_seasonality()
        self.ts_remainder()
        
    def ts_trend(self):
        self.data['timestep'] = list(range(len(self.data)))
        X = self.data[['timestep']]
        m = LinearRegression()
        m.fit(X, self.data[self.y_column])
        self.data['trend'] = m.predict(X)
        print('Trend: Linear Regressio', '\nIntercept:', m.intercept_, '\nSlope: ', m.coef_)

    def ts_seasonality(self):
        seasonal_dummies = pd.get_dummies(self.data.index.month, prefix='month')
        seasonal_dummies = seasonal_dummies.set_index(self.data.index)
        self.data = pd.concat([self.data, seasonal_dummies], axis=1)
        X = self.data.filter(regex='month', axis=1)
        m = LinearRegression()
        m.fit(X, self.data[self.y_column])
        self.data['seasonal'] = m.predict(X)

    def ts_remainder(self):
        X = self.data.filter(regex='timestep|month', axis=1)
        m = LinearRegression()
        m.fit(X, self.data[self.y_column])
        self.data['trend_seasonal'] = m.predict(X)
        self.data['remainder'] = self.data[self.y_column] - self.data['trend_seasonal']

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
