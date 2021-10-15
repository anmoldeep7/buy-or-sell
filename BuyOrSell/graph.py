import pandas as pd
import os
import json
import requests
import pendulum
pd.core.common.is_list_like = pd.api.types.is_list_like
from flask import request
from math import pi, ceil
import datetime
from bokeh.plotting import figure, show, output_file
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource
from bokeh.palettes import RdYlGn
from bokeh.transform import cumsum
import yfinance as yf
import numpy as np
import time
import tweepy
import re
# TextBlob - Python library for processing textual data
from textblob import TextBlob
# WordCloud - Python linrary for creating image wordclouds
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize  # to split sentences into words
from nltk.corpus import stopwords  # to get a list of stopwords
from collections import Counter
import math
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout



def base():
    if request.method == 'POST':
        ticker = request.form['ticker']
    return ticker

def get_days():
    yesterday =pendulum.yesterday().to_date_string()
    dt = pendulum.today()
    dw = dt.subtract(days=7).to_date_string()
    dt = dt.to_date_string()
    return {'y':yesterday, 't':dt, 'w':dw}


def get_graph_data2(ticker):

    end = datetime.datetime.now()
    start = datetime.datetime(2020,11,1)
    #yf.pdr_override()  
    #df = pdr.get_data_yahoo(ticker, start=start, end=end)
    df = pdr.DataReader(ticker, data_source='yahoo', start=start, end=end) 
    #df=pdr.DataReader("GME", "av-daily", start=datetime(2020, 2, 9),end=datetime(2020, 11, 14),api_key='WRRK9PD3JRG77B95')
    df['Date'] = pd.to_datetime(df.index)

    p = figure(x_axis_type='datetime', width = 1000, height=300, sizing_mode = "scale_width", background_fill_color='lightblue',tools="hover") 
    p.title.text = "Stock Price in the last year" 
    p.line(df.index, df.Close,color='red')
    script2, div2 = components(p)
    cdn_js = CDN.js_files
    cdn_css = CDN.css_files

    return {'script2': script2, 'div2': div2, 'cdn_js': cdn_js, 'cdn_css': cdn_css}


#------------TWITTER SENTIMENT ANALYSIS---------------
consumer_key = '80NwIsUvRQzmCnZMymiVLMbjR'
consumer_secret = 'z5yNbj20nCKiiiMuc6UhmiA685eOnbfFKQfePATqkWUmyeOIyD'
access_key = "490429267-gB0mEmhyo4mHTScUcX95LOYKCB48ElKBbjVP8SjA"
access_secret = "xi8ede3SPa5gYjECvEthwedVADaD3mkCaNwKSfi2RyuyD"


    
def initialize():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    #api = tweepy.API(auth, wait_on_rate_limit=True)
    return api
api = initialize()

def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

    result = requests.get(url).json()

    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']


def cleanUpTweet(txt):
    # Remove mentions
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    # Remove hashtags
    txt = re.sub(r'#', '', txt)
    # Remove retweets:
    txt = re.sub(r'RT : ', '', txt)
    # Remove urls
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt)
    return txt

def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

# negative, nautral, positive analysis
def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"


def get_twitter_data(ticker, company, splitted):


    first = splitted[0]
    text_query = first
    count = 500
    # Creation of query method using parameters
    #tweets = tweepy.Cursor(api.search,q=text_query,lang="en").items(count)
    tweets = tweepy.Cursor(api.search, q=company, tweet_mode='extended', lang='en', include_rts=False ,exclude_replies=True).items(count)
    df = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=['Tweet'])
    
    df['Tweet'] = df['Tweet'].apply(cleanUpTweet)
    last_tweets = df['Tweet'].head(20)
    last_tweets.to_json(orient='records')

#-------wordcloud---------

    tot=[]
    tweet = df['Tweet'].to_json(orient='records')
    for each in df['Tweet']:
        tot.append(each)
    totTuple = " ".join(tot)



    words = []
    for tweet in tot:
        tokens = word_tokenize(tweet)
        words.extend(tokens)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word)>2]
    
    
    wordcloud = WordCloud(width = 700, height = 350, 
            background_color ='white', 
            stopwords = stop_words, 
            min_font_size = 10).generate(totTuple) 
    wordcloud.to_file("BuyOrSell/static/wordcloud/cloud.png")



#-------------------------------

    df['Subjectivity'] = df['Tweet'].apply(getTextSubjectivity)
    df['Polarity'] = df['Tweet'].apply(getTextPolarity)
    df = df.drop(df[df['Tweet'] == ''].index)

    df['Score'] = df['Polarity'].apply(getTextAnalysis)
    
    #get totals
    positive = df[df['Score'] == 'Positive']
    positive = positive.count().values[0]

    neutral = df[df['Score'] == 'Neutral']
    neutral = neutral.count().values[0]

    negative = df[df['Score'] == 'Negative']
    negative = negative.count().values[0]


    x = {
        'Neutral':neutral,
        'Positive':positive,
        'Negative':negative
    }

    data = pd.Series(x).reset_index(name='value').rename(columns={'index':'score'})
    data['angle'] = data['value']/data['value'].sum() * 2*pi
    data['color'] = RdYlGn[len(x)]
    
    p = figure(sizing_mode="scale_width", title="Pie Chart",  background_fill_color='white',
            tools="hover", tooltips="@score: @value", x_range=(-0.5, 1.0))
    #toolbar_location=None,
    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='score', source=data)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None

    #word_cloud(df["Tweet"])

    script3, div3 = components(p)
    cdn_js = CDN.js_files
    cdn_css = CDN.css_files

    #Get the last 7 days info
    today = get_days()['t']
    week = get_days()['w']

    quote_for_stats = pdr.DataReader(ticker, data_source='yahoo', start=week, end=today)
    quote_for_stats = quote_for_stats.loc[:,['Close','Volume']]
    quote_for_stats['DateString']=quote_for_stats.index
    quote_for_stats['Close'] = round(quote_for_stats['Close'],2)
    shape =quote_for_stats.shape
    var = quote_for_stats['Close'][shape[0]-1] -quote_for_stats['Close'][0]
    perc = round((var/quote_for_stats['Close'][0])*100,4)
    seven = quote_for_stats 
    seven.reset_index(inplace = True)
    for i in range (shape[0]):
        seven['DateString'][i]=seven['DateString'][i].to_pydatetime()
        seven['DateString'][i]=seven['DateString'][i].strftime("%Y-%m-%d")
    seven =seven.to_dict('index')

    return {'script3': script3, 'div3': div3, 'cdn_js': cdn_js, 'cdn_css': cdn_css, 'last_tweets':last_tweets, 'ticker':ticker, 'company':company, 'seven':seven, 'perc':perc}

def get_prediction_data(ticker):
    yesterday = get_days()['y']
    today = get_days()['t']
    week = get_days()['w']

    df = pdr.DataReader(ticker, data_source='yahoo', start='2015-01-01', end=yesterday) 
    shape_df = df.shape
    #Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    #Converting the dataframe to a numpy array
    dataset = data.values

    #Get /Compute the number of rows to train the model on
    training_data_len = math.ceil( len(dataset) *.8)
    #Scale the all of the data to be values between 0 and 1 
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)

    #Create the scaled training data set 
    train_data = scaled_data[0:training_data_len  , : ]
    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train =[]
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    
    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    #Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    
    #Compiling and fitting the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train, epochs = 1, batch_size = 32)

    #Test data set
    test_data = scaled_data[training_data_len - 60: , : ]
    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    #Convert x_test to a numpy array 
    x_test = np.array(x_test)
    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    #Getting the models predicted price values
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)#Undo scaling

    #Calculate/Get the value of RMSE
    rmse= round(np.sqrt(np.mean(((predictions- y_test)**2))),2)
            
    # CONVERT TO BOKEH HERE
    #Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    
    #Get the quote
    quote_forPred = pdr.DataReader(ticker, data_source='yahoo', start='2020-01-01', end=yesterday)
    #Create a new dataframe
    new_df = quote_forPred.filter(['Close'])
    #Get teh last 60 day closing price 
    #forchart = pd.DataFrame( columns = ['Pred'])

    #count = quote.shape[0]
    count = 5
    for i in range(count):
        
        last_60_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
        last_60_days_scaled = scaler.transform(last_60_days)
    #Create an empty list
        X_test = []
    #Append the past 60 days
        X_test.append(last_60_days_scaled)
    #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
    #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
        pred_price = model.predict(X_test)
    #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        df_temp = pd.DataFrame(pred_price, columns = ['Close'])
        new_df = pd.concat([new_df,df_temp],ignore_index=True)
        
    new_df_res=new_df[-count:]    
    new_df_res = np.reshape(new_df_res, (new_df_res.shape[0],new_df_res.shape[1]))
    new_df_res['Close']=round(new_df_res['Close'],2)

    #FINAL VALUES OF QUOTE PREDICTED FOR N (count) DAYS
    values_nDays_predictions = new_df_res['Close'].values
    shape_nDays = values_nDays_predictions.shape[0]
    up_down = values_nDays_predictions[shape_nDays-1]-values_nDays_predictions[0]
    gif=''
    word = ''
    if up_down > 0:
        gif = "up_green.png"
        word = 'UP'
        
    else:
        gif  = 'down_red.png'
        word = 'DOWN'
        
    gif = 'src="static/wordcloud/'+gif+'"'

    p = figure(x_axis_type='datetime', width = 800, sizing_mode = "scale_width", background_fill_color='white', title="Stock Closing Prices Vs Trained Model") 
    p.grid.grid_line_alpha=0.3
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price $'
    p.line(valid.index, valid.Close, color='lightblue', legend_label='Actual')
    p.line(valid.index, valid.Predictions, color='red', legend_label='Predicted')
    script4, div4 = components(p)
    cdn_js = CDN.js_files
    cdn_css = CDN.css_files

    





    
    return {'rmse': rmse, 'prediction': values_nDays_predictions,'script4': script4, 'div4': div4, 'cdn_js': cdn_js, 'cdn_css': cdn_css, 'gif':gif, 'word':word}



