# Machine Learning Project 4

# Project Description

Creating an interactive webpage where an investor may look up a prediction of future closing stock prices based on a ticker symbol. The prediction is created using deep learning models and LSTM neural networks.

# Datasets to Be Used
-Yahoo Finance

-Nasdaq

# Tasks
- Data pull:
Web scraping model allowed us to find historical stock price openings and closings and earnings data.
- ETL: 
* checked and removed null values
* created a dataframe using only the close stock price column
* converted the dataframe to a numpy array to train the LSTM Model
* normalized the data before model fitting using MinMaxScaler
The training dataset created with closing price values with 180 time-steps. 
To prevent overfitting we added four hidden layers
- Database Management: 
Using SQL lite to create our database for Machine Learning and Mongo for the Flask App
- Machine Learning:
* used yfinance to pull data according to the different tickers selected from January 2010 to YTD.
* plotted the data using matplotlib for stock price history and stock volume history.
* tried three different models: Linear Regression, LSTM and ARIMA. We split the data to train and test. We made the predictions based on the historical data and calculated the r2 and the mean_squarred_error.
* found the ARIMA model to be the most accurate. 

![img1](/stock-prediction/images/Linear.jpg)

![img2](/stock-prediction/images/LSTM.jpg)

![img3](/stock-prediction/images/ARIMA.png)


- User Interface:
Used python flask, jQuery and html to create a user interactive dashboard.

![Screen Shot 1](/Flask_Stock_Dashboard/image_dashboard/initial_page.png)

![Screen Shot 2](/Flask_Stock_Dashboard/image_dashboard/mongodb_stockdata.png)

![Screen Shot 3](/Flask_Stock_Dashboard/image_dashboard/after_refresh_1.png)


# Conclusions
Model predicts close prices accurately as we checked our predictions a few days after we first run the model. 

Because external outliers were not tracked we cannot take those into accounts, for example Tweets, Reddit threads, etc. 

R2 score for Linear Regression: 0.98

R2 score for LSTM Model: 0.97

R2 score for ARIMA Model: 0.99







