# Time Series Analysis
A project using Auto Regressive Model for a short term air temperature forecast.
Analyzing the time seris and splitting it into its components:
- Trend
- Seasonality
- Remainder

Trend and Seasonality are modelled using a linear model, Remainder using a autoregressive model.

- TimeSeries.py:
   - Seperating Time Series into 3 compenents:
      - Trend
      - Seasonality
      - Remainder

   - Modelling 
      - Auto Regressive Model
      - forecasting
      - Crossvalidation

- utils.py:
   - EDA
   - plotting
   - imputing missing values
   - aggregation of time series
   - train test split

### Requirements
- Python 3.9
- pandas 1.5.2
- sklearn 1.2
- statsmodels 0.13
- plotly 5.9.0
- seaborn 0.11

### Usage
- run main.py in an interactive window to make us of the cells

