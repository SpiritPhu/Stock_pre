#import library

import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU

from itertools import cycle

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ! pip install plotly
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

from vnstock import *
from datetime import datetime, time, timedelta, timezone
yesterday = datetime.now() - timedelta(1)
yesterday = datetime.strftime(yesterday, '%Y-%m-%d')

df =  stock_historical_data(symbol='LPB',
                               start_date="2010-01-01",
                                end_date = yesterday)
# Rename columns
df.rename(columns={"TradingDate":"date","Open":"open","High":"high",\
                   "Low":"low","Close":"close", 'Volume' : 'Volume'}, inplace= True)

df.dropna(inplace=True)
df.isna().any()
df['date'] = pd.to_datetime(df['date'], format = '%Y%m%d')

from model_svr import svr_model
from model_rf import rf_model
from model_knn import knn_model
from model_lstm import lstm_model

svrdf = svr_model(df)
rfdf = rf_model(df)
knndf = knn_model(df)
lstmdf = lstm_model(df)

next_15days_df = []
for x in range(1,16):
    next_15days_ = datetime.now() + timedelta(x)
    next_15days_df.append(datetime.strftime(next_15days_, '%Y-%m-%d'))
next_15days =pd.DataFrame(next_15days_df,columns=['dates'])
full_dates = df['date'].append(next_15days['dates'])

finaldf = pd.DataFrame({
    # 'date' : full_dates,
    'svr':svrdf,
    'rf':rfdf,
    'lstm':lstmdf,
    'KNN':knndf,
})
finaldf['predict_avg'] = finaldf.mean(axis=1)
finaldf.tail(10)

# Conclusion Chart

names = cycle(['SVR', 'rf', 'LSTM', 'KNN', 'predict_avg'])

fig = px.line(finaldf[1000:], x=full_dates[1000:], y=[finaldf['svr'][1000:], finaldf['rf'][1000:],
                                          finaldf['lstm'][1000:],
                                          finaldf['KNN'][1000:],
                                          finaldf['predict_avg'][1000:]],
             labels={'x': 'Timestamp','value':'Stock close price'})
fig.update_layout(title_text='Final stock analysis chart', font_size=15, font_color='black',legend_title_text='Algorithms')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()

finaldf.tail(10)



full_dates = pd.DataFrame(full_dates)
full_dates = full_dates.rename(columns = {'0' : 'date'})
finaldf = finaldf.reset_index()
full_dates = full_dates.reset_index(drop=True)
full_dates = full_dates.reset_index()
a = pd.merge(full_dates, finaldf, how = 'left', on = 'index')
a = a.rename(columns = {0: 'date'})
#####################################
next_15days_df = []
for x in range(1, 16):
    next_15days_ = datetime.now() + timedelta(x)
    next_15days_df.append(datetime.strftime(next_15days_, '%Y-%m-%d'))
next_15days = pd.DataFrame(next_15days_df, columns=['dates'])
full_dates = df['date'].append(next_15days['dates'])
full_dates = pd.DataFrame(full_dates)
full_dates = full_dates.rename(columns={'0': 'date'})
full_dates = full_dates.reset_index(drop=True)
full_dates = full_dates.reset_index()