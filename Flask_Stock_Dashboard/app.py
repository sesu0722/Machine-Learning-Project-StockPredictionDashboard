from flask import Flask, request, render_template, jsonify
import yfinance as yf
from pymongo import MongoClient
import json
from bson import json_util
from bson.json_util import dumps
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import pandas as pd
import time


from flask import Response
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


from statsmodels.tsa.arima.model import ARIMA
import math
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pytz
import numpy as np

import model
from model import ARIMA_model,get_historical
import simplejson
# instantiate the Flask app.
app = Flask(__name__)

# This is the / route, or the main landing page route.
@app.route("/")
def home():
    # we will use Flask's render_template method to render a website template.
    return render_template("index.html")



# API Route for pulling the stock quote
@app.route("/quote")
def display_quote():
    # get a stock ticker symbol from the query string
    # default to AAPL
    symbol = request.args.get('symbol')

    # pull the stock quote
    quote = yf.Ticker(symbol)

    #return the object via the HTTP Response
    return jsonify(quote.info)

# API route for pulling the stock history
@app.route("/history")
def display_history():
    #get the query string parameters
    symbol = request.args.get('symbol')
    period = request.args.get('period', default="1y")
    interval = request.args.get('interval', default="1mo")

    #pull the quote
    quote = yf.Ticker(symbol)	
    #use the quote to pull the historical data from Yahoo finance
    hist = quote.history(period=period, interval=interval)
    #convert the historical data to JSON
    data = hist.to_json()
    #return the JSON in the HTTP response
    return data


#@app.route("/modeldata")  

#def finalmodel():
    #model
    #ticker =request.args.get('symbol', default="AAPL")
    #df_ticker = get_historical(ticker)
    #df_arima_pred = pd.DataFrame()
    #df_arima_pred = pd.DataFrame()
    #df_arima_pred= ARIMA_model(df_ticker, ticker)
   



@app.route("/modeldata")

def passfunction():
    model
    symbol=request.args.get('symbol', default="AAPL")

    ticker=symbol
    #try:
        #get_historical(ticker)
    #except:
        #return render_template('index.html',not_found=True)
    #else:
    #ticker_name=path(ticker)

    df=get_historical(ticker)
    df1=df.to_json()
    #getmodel()

    #return df1
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017
    DBS_NAME = 'stockDB'
    COLLECTION = 'stocks_now'
    FIELDS = {'Date': True, 'open': True, 'high': True, 'low': True, 'close': True,'adjclose':True,'volume':True, '_id': False}
    
    connection=  MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DBS_NAME][COLLECTION]
    data = collection.find(projection=FIELDS)
    json_combined = []
    for item in data:
        json_combined.append(item)
    json_combined = json.dumps(json_combined, default=json_util.default)
    
    #return json_combined

    df_arima=pd.DataFrame(list(collection.find()))

    mse,mae,rmse,accuracy,arima_forecast_df =ARIMA_model(ticker,df_arima)

    
    return render_template('index.html',symbol=ticker,mse=round(mse,2),mae=round(mae,2),rmse=round(rmse,2),accuracy=round(accuracy,2),arima_forecast_df=arima_forecast_df)
    #return arima_forecast_df
    #ARIMAmodel=getmodel()
    #rmse,accuracy,arima_forecast_df   
    #return render_template('index.html',rmse=round(rmse,2),accuracy=round(accuracy,2),arima_forecast_df=arima_forecast_df)
    
        
    #df_ticker= passfunction(ticker)
    #df_ticker=df_ticker.to_json()
    #return render_template('index.html',df_ticker)

if __name__ == "__main__":
    app.run(debug=True)
