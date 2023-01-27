#!/usr/bin/env python
# coding: utf-8

# In[57]:


#pip install dataframe-image


# In[79]:


import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
#import psycopg2
#import requests
#import csv
#import os
from datetime import datetime, timedelta
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pytz
from pandas.plotting import table 
# In[80]:
import dataframe_image as dfi

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import math
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from keras.models import Model
#from keras.models import model_from_json

# SQL Alchemy
from sqlalchemy import create_engine
#from config import username, password

import warnings
warnings.filterwarnings('ignore')

#import pickle as pkl
import yfinance as yf
from flask import request
import pymongo
#import app
#from app import display_quote
plt.style.use('fivethirtyeight')


# In[81]:

def get_historical(ticker):
   start = "2010-01-01"
   #ticker = 'AAPL'
   import yfinance as yf
   data = yf.download(ticker, start = start, period = "ytd")
   data = data.reset_index(level=0)
   df_ticker= data.drop_duplicates()
   df_ticker= df_ticker.dropna()
   df_ticker = df_ticker.rename(columns={
      'Date': 'Date',  
      'Open': 'open',
      'High': 'high',
      'Low': 'low',
      'Close': 'close',
      'Adj Close': 'adjclose',
      'Volume': 'volume'
      })
   df_ticker.to_csv('modeldata.csv', index=False)


   client = pymongo.MongoClient("mongodb://localhost:27017")
   db =client.stockDB
   collection = db.stocks_now
   collection.delete_many({})
   data2=df_ticker.to_dict(orient='records')
   collection.insert_many(data2)
     

   
   # Create Engine for project4 data
   #import sqlite3 as sl
   #conn=create_engine("sqlite:////data/stocks.db")

   #uploading to respecive tables in project2 database
   #df_ticker.to_sql(ticker, con=conn, if_exists='replace', index=False)

   return df_ticker

# In[84]:


def ARIMA_model(ticker,df_arima):

    ## Add a dummy row at the end. This will not be used to predict. 
    useast = datetime.now(pytz.timezone('America/New_York'))
    useast = useast.strftime('%Y-%m-%d')
    useast = datetime.strptime(useast, '%Y-%m-%d')
    first_forecast_date = useast +timedelta(1)
    
    #loop to add seven rows of dummy data
    for i in range(7):
        df_arima.loc[len(df_arima)]=df_arima.loc[len(df_arima)-1]
        next_day = useast +timedelta(i+1)
        df_arima.iloc[-1, df_arima.columns.get_loc('Date')] = next_day
        df_arima['Date'] = pd.to_datetime(df_arima["Date"], utc=True).dt.date

    df_arima = df_arima.set_index('Date')
    df_arima['avg'] = (df_arima['high'] + df_arima['low']) / 2

    train_data, test_data = df_arima[0:int(len(df_arima)*0.8)], df_arima[int(len(df_arima)*0.8):]
       
    plt.figure(figsize=(12,6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Close Prices')
    title_arima = ticker+ ' Stock Price Prediction by ARIMA'
    plt.title(title_arima)
    plt.plot(train_data['close'], 'green', label='Train data')
    plt.plot(test_data['close'], 'blue', label='Test data')
    plt.legend()
    plt.savefig('static/image/arima_train_test.png')
    #plt.show()
        
    train_arima = train_data['avg']
    test_arima = test_data['close']
    history = [x for x in train_arima]
    y = test_arima
    # make first prediction
    predictions = list()
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(y[0])
    # rolling forecasts
    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(1,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        # invert transformed prediction
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)

    # report performance
    print("--------------------------------------------------------------------")
    mse = mean_squared_error(y, predictions)
    print('Mean Squared Error(MSE): '+str(mse))
    mae = mean_absolute_error(y, predictions)
    print('Mean Absolute Error(MAE): '+str(mae))
    rmse = math.sqrt(mean_squared_error(y, predictions))
    print('Root mean square error(RMSE): '+str(rmse))
    accuracy = r2_score(y, predictions)
    print('Accuracy:'+str(accuracy))
    
    df_pred = pd.DataFrame(test_data.index)
    df_pred['predictions'] = predictions
    
    #filtering the forecasted stock price for the next 7 future days
    arima_forecast_df = df_pred.loc[(df_pred['Date'] >= first_forecast_date.date())]
    arima_forecast_df = arima_forecast_df.set_index('Date')

    dfi.export(arima_forecast_df, 'static/image/predictable.png')
    
    print("--------------------------------------------------------------------")
    plt.figure(figsize=(12,6))
    plt.plot(train_data.index[-600:], train_data['close'].tail(600), color='green', label = 'Train Stock Price')
    plt.plot(test_data.index, y, color = 'red', label = 'Test Stock Price')
    plt.plot(test_data.index, predictions, color = 'blue', label = 'Predicted Stock Price')
    plt.plot(arima_forecast_df['predictions'], color='Yellow', label = 'Forecasted Stock Price')
    title_arima = ticker+' Stock Price Prediction - ARIMA Model'
    plt.title(title_arima)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/selinasu/Desktop/Machine-Learning-Project4/Flask_Stock_Dashboard/static/image/arima_train_test_pred_forecast.png')
    #plt.show()
    print("--------------------------------------------------------------------")
    
    # Visualising the results
    plt.figure(figsize=(12,6))
    #plt.plot(train_data.index[-600:], train_data['close'].tail(600), color='green', label = 'Train Stock Price')
    plt.plot(test_data.index, y, color = 'red', label = 'Test Stock Price')
    plt.plot(test_data.index, predictions, color = 'blue', label = 'Predicted Stock Price')
    plt.plot(arima_forecast_df['predictions'], color='Yellow', label = 'Forecasted Stock Price')
    title_arima = ticker+' Stock Price Prediction - ARIMA Model'
    plt.title(title_arima)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/selinasu/Desktop/Machine-Learning-Project4/Flask_Stock_Dashboard/static/image/arima_test_pred_forecast.png')
    #plt.show()
    print("--------------------------------------------------------------------")
    return mse,mae,rmse,accuracy,arima_forecast_df 
        


# In[86]:

#def path(ticker_name):
    #return ticker_name

#df_ticker=get_historical(ticker_name)
#arima_forecast_df =ARIMA_model(ticker_name,df_ticker)
#arima_forecast_df
    #return df_ticker
#ARIMA_model(df_ticker_data)



# In[76]:



#df_pred = pd.DataFrame()
#df_pred = ARIMA_model(df_ticker)


# In[ ]:




