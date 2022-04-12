import json, urllib.request, os, random, helper
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, KBinsDiscretizer
import tensorflow_probability as tfp
from keras import backend as K

pd.set_option('display.max_columns', 50)
np.set_printoptions(threshold=np.inf)

# List of example URLS
URL_hourly_month = 'https://energy-charts.info/charts/price_spot_market/raw_data/de/month_2019_01.json'  # January 2019, hourly data
URL_hourly_year = 'https://energy-charts.info/charts/price_spot_market/raw_data/de/year_2019.json'  # year 2019, hourly data
URL_hourly_year_18 = 'https://energy-charts.info/charts/price_spot_market/raw_data/de/year_2018.json'  # year 2019, hourly data

URL_q_hourly_month = 'https://energy-charts.info/charts/price_spot_market/raw_data/de/month_15min_2019_01.json'  # January 2019, quarter hourly data
URL_q_hourly_year = 'https://energy-charts.info/charts/price_spot_market/raw_data/de/year_15min_2019.json'  # year 2019, quarter hourly data
URL_q_hourly_year_18 = 'https://energy-charts.info/charts/price_spot_market/raw_data/de/year_15min_2018.json'  # year 2019, quarter hourly data

URL_q_hourly_production_wind = 'https://energy-charts.info/charts/power_scatter/raw_data/de/year_wind_ante_post_2019.json'
URL_q_hourly_production_wind_18 = 'https://energy-charts.info/charts/power_scatter/raw_data/de/year_wind_ante_post_2018.json'

URL_q_hourly_production_solar = 'https://energy-charts.info/charts/power_scatter/raw_data/de/year_solar_ante_post_2019.json'
URL_q_hourly_production_solar_18 = 'https://energy-charts.info/charts/power_scatter/raw_data/de/year_solar_ante_post_2018.json'

"set the random seed values to obtain same results"
def setseed():
    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    print("RANDOM SEEDS RESET")

def download_forecast_real_production(URL):
    json_file = json.loads(urllib.request.urlopen(URL).read())
    time = np.array(json_file[0]['values'], dtype='O')[:, 0]
    time = np.array([datetime.utcfromtimestamp(val / 1000).strftime('%Y-%m-%d %H:%M') for val in time])
    actual = [val[1] for val in json_file[0]['values']]
    forecasted = [val[2] for val in json_file[0]['values']]
    data = [actual, forecasted]
    labels = [json_file[0]['xAxisLabel'][0]['en'], json_file[0]['y1AxisLabel'][0]['en']]
    df = pd.DataFrame(data).T
    df.index = pd.DatetimeIndex(time)
    df.columns = labels
    return df


def download_prices(URL, DA_bool=True):
    """
    Function to download data from energy-charts.info. In particular, the price data of the day-ahead auction and the continous intraday market are downloaded.
    """
    json_file = json.loads(urllib.request.urlopen(URL).read())

    if DA_bool:  # "Day Ahead Auction" (hourly) prices are on index 5
        print(json_file[5]['key'][0]['en'])
        Prices = np.array(json_file[5]['values'], dtype='O')
    else:  # "Intraday Continuous Average Price" (quarter-hourly) prices are on index 7
        print(json_file[7]['key'][0]['en'])
        Prices = np.array(json_file[7]['values'], dtype='O')

    # get dates
    dates = np.unique(np.array([datetime.utcfromtimestamp(val[0] / 1000).strftime('%Y-%m-%d %H:%M') for val in Prices]))

    # embed days into lists
    Prices = [[val[1] for val in Prices if datetime.fromtimestamp(val[0] / 1000).date() == date] for date in dates]
    return Prices, dates


def get_pandas_df(URL):
    """
        returns pandas datafram with all time series in the json file
    """
    json_file = json.loads(urllib.request.urlopen(URL).read())
    labels = [json_file[i]['key'][0]['en'] for i in range(12)]
    # retrieve time information
    time = np.array(json_file[0]['values'], dtype='O')[:, 0]
    time = np.array([datetime.utcfromtimestamp(val / 1000).strftime('%Y-%m-%d %H:%M') for val in time])
    data = np.array([np.array(json_file[i]['values'], dtype='O')[:, 1] for i in range(12)])
    df = pd.DataFrame(data.T, index=time)
    df.columns = labels
    return df.astype('float32')

"""
Convert the price column into 4 columns, each column corresponding to quarter hours
"""
def getMultiDim(colname, data, newnames):
    Q1 = data.loc[data.index.minute == 00, colname].rename(newnames[0])
    Q2 = data.loc[data.index.minute == 15, colname].rename(newnames[1])
    Q3 = data.loc[data.index.minute == 30, colname].rename(newnames[2])
    Q4 = data.loc[data.index.minute == 45, colname].rename(newnames[3])
    Q1.index = Q1.index.floor('H')
    Q2.index = Q2.index.floor('H')
    Q3.index = Q3.index.floor('H')
    Q4.index = Q4.index.floor('H')
    multi_dimensional_data = pd.concat([Q1, Q2, Q3, Q4], axis=1)
    return multi_dimensional_data


def setindex(data, start, period, freq, *args):
    data.index = pd.to_datetime(data.index)
    data.index = data.index + pd.Timedelta('1 hour')
    indexes = pd.date_range(start, periods=period, freq=freq)
    if data.shape[0] != period:
        data = data.reindex(indexes, fill_value=0)
    # else: data = data.set_index(indexes)

    return data

"drop the observations earlier and later than the specified dates "
def drop_values(data, drop_after='2019-12-31', drop_before='2018-01-01 23:45'):
    data = data.loc[drop_before < data.index]
    data = data.loc[data.index < drop_after]
    return data


# Add hour and quarter
def add_time_features(data):
    data['Hour'] = data.index.hour
    data['Day'] = data.index.weekday
    data['Month'] = data.index.month
    return data


"""
concatenate additional features to the price data
"""
def add_features_from_other_df(intraday, other):
    intraday = pd.concat([intraday, other], axis=1)
    return intraday

"""
Take difference between two columns, needed to obtain price differences and forecast errors
"""
def get_delta_features(data, col):
    data = data[col[0]] - data[col[1]]
    return data

"""
The realnvp needs to have the distribution variables in the first columns of the dataframe.
This functions ensures that
"""
def set_the_column_order(cols, data):
    ordered_cols = cols + [col for col in data.columns if col not in cols]
    ordered_data = data[ordered_cols]
    return ordered_data

"""
Load  wind and solar actual generation and name the columns
"""
def get_wind_solar_actual(path):
    df = pd.read_csv(path)
    df = df[['MTU', 'Wind Onshore  - Actual Aggregated [MW]',
             'Wind Offshore  - Actual Aggregated [MW]', 'Solar  - Actual Aggregated [MW]']]
    df[["datetime", "drop"]] = df['MTU'].str.split('-', 1, expand=True)
    df['wind_actual'] = df['Wind Onshore  - Actual Aggregated [MW]'] + df['Wind Offshore  - Actual Aggregated [MW]']
    df.index = pd.to_datetime(df['datetime'])
    df.drop(['MTU', 'drop', 'Wind Onshore  - Actual Aggregated [MW]',
             'Wind Offshore  - Actual Aggregated [MW]', 'datetime'], axis=1, inplace=True)
    df = df.rename(columns={'Solar  - Actual Aggregated [MW]': 'solar_actual'})
    return df

"""
Load  wind and solar generation forecast and name the columns
"""
def get_wind_solar_forecast(path):
    df = pd.read_csv(path)
    df = df[['MTU (UTC)', 'Generation - Wind Offshore  [MW] Day Ahead/ Germany (DE)',
             'Generation - Wind Onshore  [MW] Day Ahead/ Germany (DE)',
             'Generation - Solar  [MW] Day Ahead/ Germany (DE)']]
    df[["datetime", "drop"]] = df['MTU (UTC)'].str.split('-', 1, expand=True)
    df['wind_forecast'] = df['Generation - Wind Offshore  [MW] Day Ahead/ Germany (DE)'] + \
                          df['Generation - Wind Onshore  [MW] Day Ahead/ Germany (DE)']
    df.index = pd.to_datetime(df['datetime'])

    df.drop(['MTU (UTC)', 'drop', 'Generation - Wind Offshore  [MW] Day Ahead/ Germany (DE)',
             'Generation - Wind Onshore  [MW] Day Ahead/ Germany (DE)', 'datetime'], axis=1, inplace=True)
    df = df.rename(columns={'Generation - Solar  [MW] Day Ahead/ Germany (DE)': 'solar_forecast'})
    return df

"""
Load  demand actual and forecasted generation and name the columns
"""
def get_consumption(path):
    df = pd.read_csv(path)
    df[["datetime", "drop"]] = df['Time (UTC)'].str.split('-', 1, expand=True)
    df.index = pd.to_datetime(df['datetime'])
    df.drop(["Time (UTC)", "drop", "datetime"], axis=1, inplace=True)
    df = df.rename(columns={'Day-ahead Total Load Forecast [MW] - Germany (DE)': 'load_forecast',
                            'Actual Total Load [MW] - Germany (DE)': 'load_actual'})
    return df


ID_prices_19 = get_pandas_df(URL_q_hourly_year) #get the ID prices -2019
DA_prices_19 = get_pandas_df(URL_hourly_year) #get the DA prcies - 2019
ID_prices_18 = get_pandas_df(URL_q_hourly_year_18)#get the ID prices -2018
DA_prices_18 = get_pandas_df(URL_hourly_year_18)#get the DA prices -2018

ID_prices_18 = ID_prices_18[:-1]
ID_prices = pd.concat([ID_prices_18, ID_prices_19]) 
DA_prices = pd.concat([DA_prices_18, DA_prices_19])

wind_solar_actual_18 = get_wind_solar_actual("../data/Actual Generation per Production Type_2018-01-01-2019-01-01.csv")
wind_solar_actual_19 = get_wind_solar_actual("../data/Actual Generation per Production Type_2019-01-01-2020-01-01.csv")
wind_solar_actual = pd.concat([wind_solar_actual_18, wind_solar_actual_19])
wind_solar_forecast_18 = get_wind_solar_forecast(
    "../data/Generation Forecasts for Wind and Solar_2018-01-01-2019-01-01.csv")
wind_solar_forecast_19 = get_wind_solar_forecast(
    "../data/Generation Forecasts for Wind and Solar_2019-01-01-2020-01-01.csv")
wind_solar_forecast = pd.concat([wind_solar_forecast_18, wind_solar_forecast_19])
wind_solar = pd.concat([wind_solar_forecast, wind_solar_actual], axis=1)

consumption_18 = get_consumption("../data/Total Load - Day Ahead _ Actual_2018-01-01-2019-01-01.csv")
consumption_19 = get_consumption("../data/Total Load - Day Ahead _ Actual_2019-01-01-2020-01-01.csv")

consumption = pd.concat([consumption_18, consumption_19])

""" set the indexes from 1.01.2018 to 32.12.2019"""
ID_prices = setindex(data=ID_prices, start='2018-01-01', period=ID_prices.shape[0], freq='15T')
DA_prices = setindex(data=DA_prices, start='2018-01-01', period=DA_prices.shape[0], freq='60T')

wind_solar = setindex(data=wind_solar, start='2018-01-01', period=wind_solar.shape[0], freq='15T')
consumption = setindex(data=consumption, start='2018-01-01', period=consumption.shape[0], freq='15T')

"""  drop the data points with missing values """
ID_prices = drop_values(ID_prices, drop_after='2019-12-31', drop_before='2018-01-01 23:45')

DA_prices = drop_values(DA_prices, drop_after='2019-12-31', drop_before='2018-01-01 23:45')

wind_solar = drop_values(wind_solar, drop_after='2019-12-31', drop_before='2018-01-01 23:45')

consumption = drop_values(consumption, drop_after='2019-12-31', drop_before='2018-01-01 23:45')

""" convert the one dimensional data into 4 dimensions"""
ID3_quarterly = getMultiDim('Intraday Continuous 15 minutes ID3-Price', ID_prices, ['Q1', 'Q2', 'Q3', 'Q4'])
Coventional_quarterly = getMultiDim('Conventional > 100 MW', ID_prices,['Conventional Q1', 'Conventional Q2',
                                                         'Conventional Q3', 'Conventional Q4'])
wind_solar['wind_forecast_error'] = get_delta_features(wind_solar,
                                                       ['wind_actual', 'wind_forecast'])
wind_solar['solar_forecast_error'] = get_delta_features(wind_solar,
                                                        ['solar_actual', 'solar_forecast'])
consumption['load_forecast_error'] = get_delta_features(consumption,
                                                        ['load_actual', 'load_forecast'])
wind_production_quarterly = getMultiDim('wind_actual', wind_solar,
                                        ['Wind_Q1_actual', 'Wind_Q2_actual', 'Wind_Q3_actual', 'Wind_Q4_actual'])
solar_production_quarterly = getMultiDim('solar_actual', wind_solar,
                                         ['Solar_Q1_actual', 'Solar_Q2_actual', 'Solar_Q3_actual', 'Solar_Q4_actual'])
wind_forecast_quarterly = getMultiDim('wind_forecast', wind_solar,
                                      ['Wind_Q1_forecast', 'Wind_Q2_forecast', 'Wind_Q3_forecast', 'Wind_Q4_forecast'])
solar_forecast_quarterly = getMultiDim('solar_forecast', wind_solar,
                                       ['Solar_Q1_forecast', 'Solar_Q2_forecast', 'Solar_Q3_forecast',
                                        'Solar_Q4_forecast'])

wind_production_err_quarterly = getMultiDim('wind_forecast_error', wind_solar,
                                            ['Wind_Q1_error', 'Wind_Q2_error', 'Wind_Q3_error', 'Wind_Q4_error'])
solar_production_err_quarterly = getMultiDim('solar_forecast_error', wind_solar,
                                             ['Solar_Q1_error', 'Solar_Q2_error', 'Solar_Q3_error', 'Solar_Q4_error'])
consumption_error_quarterly = getMultiDim('load_forecast_error', consumption,
                                          ['load_Q1_error', 'load_Q2_error', 'load_Q3_error', 'load_Q4_error'])
consumption_forecast_quarterly = getMultiDim('load_forecast', consumption,
                                             ['load_Q1_forecast', 'load_Q2_forecast', 'load_Q3_forecast',
                                              'load_Q4_forecast'])
consumption_actual_quarterly = getMultiDim('load_actual', consumption,
                                           ['load_Q1_actual', 'load_Q2_actual', 'load_Q3_actual',
                                            'load_Q4_actual'])
"""
Append the corresponding day-ahead prices to the ID price data
"""
ID3_quarterly = add_features_from_other_df(ID3_quarterly, DA_prices['Day Ahead Auction'])
"""
Append the corresponding conventional production to the ID price data
"""
ID3_quarterly = add_features_from_other_df(ID3_quarterly, DA_prices['Conventional > 100 MW'])
ID3_quarterly = ID3_quarterly.rename(columns={'Conventional > 100 MW': 'Conventional'})

"""
Create delta prices
"""
for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
    ID3_quarterly['DA_' + quarter] = get_delta_features(ID3_quarterly, ['Day Ahead Auction', quarter])

"""
Add contiditioning variables to price delta data
"""
ID3_quarterly = add_features_from_other_df(ID3_quarterly, Coventional_quarterly)

ID3_quarterly = add_features_from_other_df(ID3_quarterly, wind_production_quarterly)
ID3_quarterly = add_features_from_other_df(ID3_quarterly, wind_forecast_quarterly)
ID3_quarterly = add_features_from_other_df(ID3_quarterly, wind_production_err_quarterly)

ID3_quarterly = add_features_from_other_df(ID3_quarterly, solar_production_quarterly)
ID3_quarterly = add_features_from_other_df(ID3_quarterly, solar_forecast_quarterly)
ID3_quarterly = add_features_from_other_df(ID3_quarterly, solar_production_err_quarterly)
ID3_quarterly = add_features_from_other_df(ID3_quarterly, consumption_error_quarterly)
ID3_quarterly = add_features_from_other_df(ID3_quarterly, consumption_forecast_quarterly)
ID3_quarterly = add_features_from_other_df(ID3_quarterly, consumption_actual_quarterly)

ID3_quarterly = add_time_features(ID3_quarterly)

ID3_quarterly_EDA = ID3_quarterly.copy()
"""
drop na values if there are any
"""
ID3_quarterly.dropna(inplace=True)

"""
Drop the ID3 prices
"""
ID3_quarterly.drop(['Q1', 'Q2', 'Q3', 'Q4'], axis=1, inplace=True)

""" 
Order the columns in a way that the price deltes are in the first columns
"""
ID3_quarterly = set_the_column_order(['DA_Q1', 'DA_Q2', 'DA_Q3', 'DA_Q4'], ID3_quarterly)


"""
Drop outliers
"""
ID3_quarterly = ID3_quarterly.loc[
    (ID3_quarterly['DA_Q1'] > -150)  & (ID3_quarterly['DA_Q2'] > -190) & (ID3_quarterly['DA_Q3'] > -100) & (
            ID3_quarterly['DA_Q4'] > -150) & (ID3_quarterly['DA_Q2'] < ID3_quarterly['DA_Q2'].max()) & (
                ID3_quarterly['DA_Q3'] < ID3_quarterly['DA_Q3'].max())& (
                ID3_quarterly['DA_Q4'] < ID3_quarterly['DA_Q4'].max())]
"""
Split the data into train and test sets and then scale each column
"""
class FeatureEngineering():
    def __init__(self, prices, numeric_features, to_be_categorized_features, categorical_features, cyclical_features,
                 test_conditions):
        self.prices = prices
        self.numeric_features = numeric_features
        self.to_be_categorized_features = to_be_categorized_features
        self.categorical_features = categorical_features
        self.cyclical_features = cyclical_features
        self.test_conditions = test_conditions
    """train test split"""
    def get_train_test(self, data):
        columns = self.prices.copy()
        columns.extend(self.numeric_features)
        columns.extend(self.to_be_categorized_features)
        columns.extend(self.categorical_features)
        columns.extend(self.cyclical_features)

        # train = data.loc[data.index<'2019-11-01 00:00:00', columns] #train set
        train = data.sample(frac=0.85)
        test = data[columns].drop(train.index) #test set
        """
        If the user wants to get a test set with specific conditioning variables, test_with_conditions 
        extracts the observations with sepcified condtidions
        """
        qry = ' and '.join(["{} == {}".format(k, v) for k, v in self.test_conditions.items()])
        if qry != '':
            test_with_conditions = test.query(qry)
        else:
            test_with_conditions = test
        return train, test, test_with_conditions
    """ Scale the numerical columns, encode cyclical features trigonometric and categorize numerical 
        features if there is any """
    def transformers(self, train):
        mean = train[self.prices].values.mean()
        std = train[self.prices].values.std(ddof=1)
        column_transformer = ColumnTransformer(
            [('Scaler_prices', StandardScaler(with_mean=mean, with_std=std), self.prices),
             ('Scalers_numeric', StandardScaler(), self.numeric_features),
             ('OneHot', OneHotEncoder(sparse=False, drop='first'), self.categorical_features),
             ('Bin', KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='kmeans'),
              self.to_be_categorized_features)],
            remainder='drop')
        sin_transformer = {}
        cos_transformer = {}
        for cycle in self.cyclical_features:
            if cycle == 'Hour':
                period = 24
            elif cycle == 'Day':
                period = 7
            elif cycle == 'Month':
                period = 12
            else:
                period = 1
            sin_transformer[cycle] = (FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi)))
            cos_transformer[cycle] = (FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi)))
        return column_transformer, sin_transformer, cos_transformer
    """fit the transformers"""
    def fit(self, train):
        self.column_transformer, self.sin_trans, self.cos_trans = self.transformers(train)
        self.column_transformer.fit(train)
    """transform the data"""
    def transform(self, test):
        test_ = pd.DataFrame(data=self.column_transformer.transform(test), index=test.index,
                            columns=helper.get_feature_names(self.column_transformer))
        for key in self.sin_trans:
            test_[key + '_sin'] = self.sin_trans[key].fit_transform(test[key].values)
            test_[key + '_cos'] = self.cos_trans[key].fit_transform(test[key].values)
        return test_
""" apply feature engineering to the price data and specified additional information
    create train test sets for each of the implemented models and store them in dictionaries
"""
def get_train_test_sets(dataset,
                        target_col_names=None,
                        numeric_features=None,
                        categorical_features=None,
                        to_be_categorized_features=None,
                        cyclical_features=None,
                        conditions=None,
                        n_sample=1000):
    if categorical_features is None:
        categorical_features = []
    if conditions is None:
        conditions = {}
    if cyclical_features is None:
        cyclical_features = []
    if to_be_categorized_features is None:
        to_be_categorized_features = []
    if numeric_features is None:
        numeric_features = ['Wind_Q1_error', 'Wind_Q2_error', 'Wind_Q3_error', 'Wind_Q4_error',
                            'Solar_Q1_error', 'Solar_Q2_error', 'Solar_Q3_error', 'Solar_Q4_error',
                            'load_Q1_forecast', 'load_Q2_forecast', 'load_Q3_forecast', 'load_Q4_forecast',
                            'Conventional']
    if target_col_names is None:
        target_col_names = ['DA_Q1', 'DA_Q2', 'DA_Q3', 'DA_Q4']
    preprocess = FeatureEngineering(target_col_names, numeric_features, to_be_categorized_features,
                                    categorical_features,
                                    cyclical_features, conditions)
    train, test, _ = preprocess.get_train_test(dataset)
    print(train.shape, test.shape)
    preprocess.fit(train)
    train = preprocess.transform(train)
    test = preprocess.transform(test)
    test.sort_index(inplace=True)
    train = train.sample(frac=1) #shuffle the data

    target_variables_train = train[target_col_names]
    exo_variables_train = train.drop(target_col_names, axis=1)

    target_variables_test = test[target_col_names]
    exo_variables_test = test.drop(target_col_names, axis=1)
    training_datasets = {'unconditional realnvp': target_variables_train, 
                         'conditional realnvp': train,
                         'unconditional guassian': [''],
                         'conditional gaussian': [exo_variables_train, target_variables_train],
                         'copula LQR': [exo_variables_train, target_variables_train],
                         'copula ANN': [exo_variables_train, target_variables_train]}

    distribution = tfp.distributions.MultivariateNormalDiag(loc=[0.0] * len(target_col_names), scale_diag=[1.0] * len(target_col_names))
    test_datasets = { 'full': test,
                     'real': target_variables_test, 
                     'unconditional realnvp':pd.DataFrame(data=distribution.sample(test.shape[0]*n_sample).numpy(),
                                            columns=target_col_names),
                     'conditional realnvp': pd.concat(
                         [pd.DataFrame(distribution.sample(test.shape[0] * n_sample).numpy(), columns=target_col_names),
                            pd.DataFrame(np.repeat(exo_variables_test.values, n_sample, axis=0))], axis=1),
                     'conditional gaussian': exo_variables_test,
                     'copula LQR': exo_variables_test,
                     'copula ANN': exo_variables_test}
    return training_datasets, test_datasets
