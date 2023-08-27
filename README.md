# Time Series Analysis Project

This project is focused on performing time series analysis on temperature data using Python. The analysis includes trend, seasonality, autoregressive modeling, cross-validation, and forecasting.

## Project Structure

### Files

- `TimeSeries.py`: Python script containing the `TimeSeries` class that handles various time series analysis tasks.
- `utils.py`: Utility script with functions for data cleaning, visualization, and analysis.
- `data/TG_STAID002759.txt`: Raw temperature data file in a specific format.
- `main.py`: Main script that reads the data, performs preprocessing, and runs the time series analysis.

### Analysis

The analysis workflow includes the following steps:

1. **Data Preprocessing:**
   - Read the raw temperature data from the provided file.
   - Clean the data by removing unnecessary columns and handling missing values.

2. **Exploratory Data Analysis (EDA):**
   - Generate basic statistics about the temperature data.
   - Create interactive time series plots using Plotly for better visualization.

3. **Imputation and Aggregation:**
   - Impute missing temperature values using data from nearby years.
   - Aggregate the daily temperature data to monthly values.

4. **Time Series Analysis:**
   - Train a time series model using the `TimeSeries` class.
   - Perform trend, seasonality, and autoregressive analysis.
   - Visualize the results using various plots.

5. **Model Validation:**
   - Apply the trained model to validation data to assess its performance.
   - Evaluate the model using cross-validation techniques.

6. **Forecasting:**
   - Use the trained model to make future temperature forecasts.
   - Visualize the forecasted data along with the last week of the time series.

## Usage
1. Run the `main.py` script to perform the time series analysis and generate the results. The script can be executed in an interactive terminal within Visual Studio Code.

## Requirements

- Python 3.9
- pandas 1.5.2
- scikit-learn 1.2
- statsmodels 0.13
- plotly 5.9.0
- seaborn 0.11

## Acknowledgments

The temperature data used in this project is provided by the European Climate Assessment & Dataset (ECA&D). Please refer to the source mentioned in the data file for more information about the dataset.

## License

This project is licensed under the [MIT License](LICENSE).