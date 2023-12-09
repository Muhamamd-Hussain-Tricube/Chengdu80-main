import yfinance as yf
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import tensorflow,keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Concatenate, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping
from keras.regularizers import L1L2
from keras.layers import TimeDistributed
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ta.momentum import RSIIndicator, WilliamsRIndicator, ROCIndicator, StochasticOscillator, AwesomeOscillatorIndicator, KAMAIndicator, StochRSIIndicator, UltimateOscillator, PercentagePriceOscillator
from ta.volume import ChaikinMoneyFlowIndicator, MFIIndicator, OnBalanceVolumeIndicator, EaseOfMovementIndicator, VolumePriceTrendIndicator, NegativeVolumeIndexIndicator, VolumeWeightedAveragePrice
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel, KeltnerChannel, UlcerIndex
from ta.trend import MACD, AroonIndicator, CCIIndicator, EMAIndicator, ADXIndicator, DPOIndicator, PSARIndicator, IchimokuIndicator, KSTIndicator, STCIndicator, TRIXIndicator, VortexIndicator
from ta.others import CumulativeReturnIndicator, DailyLogReturnIndicator, DailyReturnIndicator
from termcolor import colored
from sklearn.ensemble import BaggingClassifier
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from transformers import GPT2Model, GPT2Tokenizer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
import torch
import json
import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)
np.random.seed(23904)

def read_and_parse_csv(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        content = file.readlines()

    headers = content[0].rstrip().split(',')
    data = []
    for line in content[1:]:
        fields = []
        field = ''
        in_quotes = False
        for ch in line:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == ',':
                if in_quotes:
                    field += ch
                else:
                    fields.append(field)
                    field = ''
            else:
                field += ch
        if field:
            fields.append(field.rstrip())
        data.append(dict(zip(headers, fields)))

    return data

mega_csv = read_and_parse_csv('mega.csv')
for j in range(len(mega_csv)):
    row = mega_csv[j]
    keywords_frequency_str = row['keyword_frequency']
    new_str = ''
    new_str = new_str + keywords_frequency_str[0]
    for i in range(1, len(keywords_frequency_str)):
        if keywords_frequency_str[i - 1] == '{':
            new_str = new_str + '"' + keywords_frequency_str[i]
        elif keywords_frequency_str[i] == ':':
            new_str = new_str + '":'
        elif keywords_frequency_str[i - 1] == ' ' and keywords_frequency_str[i].isalpha() and keywords_frequency_str[i - 2] == ',':
            new_str = new_str + '"' + keywords_frequency_str[i]
        else:
            new_str = new_str + keywords_frequency_str[i]
    mega_csv[j]['keyword_frequency'] = new_str
    
mega_csv = pd.DataFrame(mega_csv)
unique_tickers = mega_csv['ticker'].unique()
all_dfs = {ticker: mega_csv[mega_csv['ticker'] == ticker] for ticker in unique_tickers}

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()


# List of tickers
tickers = unique_tickers

# Dictionaries to store data
train_data = {}
valid_data = {}
test_data = {}
chmox
# Loop through tickers
for ticker in tickers:

    print(f"Processing {ticker}...")
    df = all_dfs[ticker]
    df = df.reset_index(drop=True)
    print("Preparing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    print("Embedding headlines...")
    df['headline'] = df['headline'].apply(get_embedding)

    cnt_i = 0
    for i in df['headline']:
        cnt = 0
        for j in i:
            if f'headline_{cnt}' not in df.columns:
                df[f'headline_{cnt}'] = np.nan
            df[f'headline_{cnt}'][cnt_i] = j
            cnt += 1
        cnt_i += 1
    df.drop('headline', axis=1, inplace=True)

    print("Embedding keywords...")
    df['keyword_frequency'] = df['keyword_frequency'].apply(get_embedding)

    cnt_i = 0
    for i in df['keyword_frequency']:
        cnt = 0
        for j in i:
            if f'keyword_frequency_{cnt}' not in df.columns:
                df[f'keyword_frequency_{cnt}'] = np.nan
            df[f'keyword_frequency_{cnt}'][cnt_i] = j
            cnt += 1
        cnt_i += 1
    df.drop('keyword_frequency', axis=1, inplace=True)

    print("Downloading OHLC data...")
    ohlc_data = yf.download(ticker, start=df['Date'].min(), end=df['Date'].max())
    ohlc_data.reset_index(inplace=True)
    ohlc_data['Date'] = ohlc_data['Date'].dt.strftime('%Y-%m-%d')
    df = pd.merge(df, ohlc_data, on='Date', how='inner')

    # Fill NaN values
    data = df.fillna(method='ffill')
    
    # Calculate indices for split
    train_size = int(0.6 * len(data))
    valid_size = int(0.25 * len(data))

    # Split data
    train = data[:train_size].dropna()
    valid = data[train_size:train_size + valid_size].dropna()
    test = data[train_size + valid_size:].dropna()

    # Store in dictionaries
    train_data[ticker] = train
    valid_data[ticker] = valid
    test_data[ticker] = test

    # Extract train and test data for current ticker
    train = train_data[ticker]
    valid = valid_data[ticker]
    test = test_data[ticker]
    
    # Extract features and targets
    names = ['Open', 'High', 'Low', 'Close']
    for i in range(768):
        names.append(f'headline_{i}')
    names.append('sentiment_score')
    for i in range(768):
        names.append(f'keyword_frequency_{i}')
    X_train = train[names].dropna()
    
    X_valid = valid[names].dropna()

    X_test = test[names].dropna()
    
    y_open_train = train['Open'].dropna()
    y_high_train = train['High'].dropna()
    y_low_train = train['Low'].dropna()
    y_close_train = train['Close'].dropna()
    
    y_open_valid = valid['Open'].dropna()
    y_high_valid = valid['High'].dropna()
    y_low_valid = valid['Low'].dropna() 
    y_close_valid = valid['Close'].dropna()

    y_open_test = test['Open'].dropna()
    y_high_test = test['High'].dropna()
    y_low_test = test['Low'].dropna()
    y_close_test = test['Close'].dropna()
    
    # Normalize data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test) 

    #Reshape input data for different models
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_valid_cnn = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_valid_lstm = X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[1])
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Train the CNN model

    model_open_cnn = Sequential()
    model_open_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
    model_open_cnn.add(MaxPooling1D(pool_size=2))
    model_open_cnn.add(Flatten())
    model_open_cnn.add(Dense(50, activation='relu'))
    model_open_cnn.add(Dense(1))
    model_open_cnn.compile(optimizer='adam', loss='mae')
    model_open_cnn.fit(X_train_cnn, y_open_train, epochs=50, batch_size=64)

    model_high_cnn = Sequential()
    model_high_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
    model_high_cnn.add(MaxPooling1D(pool_size=2))
    model_high_cnn.add(Flatten())
    model_high_cnn.add(Dense(50, activation='relu'))
    model_high_cnn.add(Dense(1))
    model_high_cnn.compile(optimizer='adam', loss='mae')
    model_high_cnn.fit(X_train_cnn, y_high_train, epochs=50, batch_size=64)

    model_low_cnn = Sequential()
    model_low_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
    model_low_cnn.add(MaxPooling1D(pool_size=2))
    model_low_cnn.add(Flatten())
    model_low_cnn.add(Dense(50, activation='relu'))
    model_low_cnn.add(Dense(1))
    model_low_cnn.compile(optimizer='adam', loss='mae')
    model_low_cnn.fit(X_train_cnn, y_low_train, epochs=50, batch_size=64)

    model_close_cnn = Sequential()
    model_close_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
    model_close_cnn.add(MaxPooling1D(pool_size=2))
    model_close_cnn.add(Flatten())
    model_close_cnn.add(Dense(50, activation='relu'))
    model_close_cnn.add(Dense(1))
    model_close_cnn.compile(optimizer='adam', loss='mae')
    model_close_cnn.fit(X_train_cnn, y_close_train, epochs=50, batch_size=64)

    model_open_cnn.save("model_open_cnn.h5")
    model_high_cnn.save("model_high_cnn.h5")
    model_low_cnn.save("model_low_cnn.h5")
    model_close_cnn.save("model_close_cnn.h5")

    # Create EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


    # Train the LSTM model

    model_open_lstm = Sequential()
    model_open_lstm.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model_open_lstm.add(BatchNormalization())  
    model_open_lstm.add(Conv1D(256, 2, padding='same', activation='relu'))
    model_open_lstm.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model_open_lstm.add(BatchNormalization())
    model_open_lstm.add(Conv1D(128, 2, padding='same', activation='relu'))
    model_open_lstm.add(TimeDistributed(Dense(256, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_open_lstm.add(Dropout(0.4))
    model_open_lstm.add(TimeDistributed(Dense(128, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_open_lstm.add(Dropout(0.4))
    model_open_lstm.add(TimeDistributed(Dense(64, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))  
    model_open_lstm.add(Dropout(0.4)) 
    model_open_lstm.add(TimeDistributed(Dense(1)))
    model_open_lstm.compile(loss='mae', optimizer='adam')

    model_open_lstm.fit(X_train_lstm, y_open_train, epochs=100, batch_size=64, validation_data=(X_valid_lstm, y_open_valid), callbacks=[early_stopping])

    model_high_lstm = Sequential()
    model_high_lstm.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model_high_lstm.add(BatchNormalization())
    model_high_lstm.add(Conv1D(256, 2, padding='same', activation='relu'))
    model_high_lstm.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model_high_lstm.add(BatchNormalization())  
    model_high_lstm.add(Conv1D(128, 2, padding='same', activation='relu'))
    model_high_lstm.add(TimeDistributed(Dense(256, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_high_lstm.add(Dropout(0.4))
    model_high_lstm.add(TimeDistributed(Dense(128, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_high_lstm.add(Dropout(0.4))
    model_high_lstm.add(TimeDistributed(Dense(64, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_high_lstm.add(Dropout(0.4))  
    model_high_lstm.add(TimeDistributed(Dense(1)))
    model_high_lstm.compile(loss='mae', optimizer='adam') 

    model_high_lstm.fit(X_train_lstm, y_high_train, epochs=100, batch_size=64, validation_data=(X_valid_lstm, y_high_valid), callbacks=[early_stopping])

    model_low_lstm = Sequential()
    model_low_lstm.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))  
    model_low_lstm.add(BatchNormalization())
    model_low_lstm.add(Conv1D(256, 2, padding='same', activation='relu'))
    model_low_lstm.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model_low_lstm.add(BatchNormalization())
    model_low_lstm.add(Conv1D(128, 2, padding='same', activation='relu'))
    model_low_lstm.add(TimeDistributed(Dense(256, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_low_lstm.add(Dropout(0.4))
    model_low_lstm.add(TimeDistributed(Dense(128, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_low_lstm.add(Dropout(0.4))  
    model_low_lstm.add(TimeDistributed(Dense(64, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_low_lstm.add(Dropout(0.4))
    model_low_lstm.add(TimeDistributed(Dense(1)))
    model_low_lstm.compile(loss='mae', optimizer='adam')

    model_low_lstm.fit(X_train_lstm, y_low_train, epochs=100, batch_size=64, validation_data=(X_valid_lstm, y_low_valid), callbacks=[early_stopping])

    model_close_lstm = Sequential()
    model_close_lstm.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model_close_lstm.add(BatchNormalization())
    model_close_lstm.add(Conv1D(256, 2, padding='same', activation='relu'))
    model_close_lstm.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))  
    model_close_lstm.add(BatchNormalization())
    model_close_lstm.add(Conv1D(128, 2, padding='same', activation='relu'))
    model_close_lstm.add(TimeDistributed(Dense(256, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_close_lstm.add(Dropout(0.4)) 
    model_close_lstm.add(TimeDistributed(Dense(128, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_close_lstm.add(Dropout(0.4))
    model_close_lstm.add(TimeDistributed(Dense(64, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))))
    model_close_lstm.add(Dropout(0.4))  
    model_close_lstm.add(TimeDistributed(Dense(1)))
    model_close_lstm.compile(loss='mae', optimizer='adam')

    model_close_lstm.fit(X_train_lstm, y_close_train, epochs=100, batch_size=64, validation_data=(X_valid_lstm, y_close_valid), callbacks=[early_stopping])

    model_open_lstm.save("model_open_lstm.h5")
    model_high_lstm.save("model_high_lstm.h5")
    model_low_lstm.save("model_low_lstm.h5")
    model_close_lstm.save("model_close_lstm.h5")

    # Train RF Model
    model_open_rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=0)
    model_open_rf.fit(X_train, y_open_train)
    model_high_rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=0)
    model_high_rf.fit(X_train, y_high_train)
    model_low_rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=0)
    model_low_rf.fit(X_train, y_low_train)
    model_close_rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=0)
    model_close_rf.fit(X_train, y_close_train)

    model_open_rf.save("model_open_rf.h5")
    model_high_rf.save("model_high_rf.h5")
    model_low_rf.save("model_low_rf.h5")
    model_close_rf.save("model_close_rf.h5")
    # Train SVM Model
    model_open_svm = SVR(kernel='rbf', C=1.0, gamma='scale')
    model_open_svm.fit(X_train, y_open_train)

    model_high_svm = SVR(kernel='rbf', C=1.0, gamma='scale') 
    model_high_svm.fit(X_train, y_high_train)

    model_low_svm = SVR(kernel='rbf', C=1.0, gamma='scale')
    model_low_svm.fit(X_train, y_low_train)

    model_close_svm = SVR(kernel='rbf', C=1.0, gamma='scale')
    model_close_svm.fit(X_train, y_close_train)

    model_open_svm.save("model_open_svm.h5")
    model_high_svm.save("model_high_svm.h5")
    model_low_svm.save("model_low_svm.h5")
    model_close_svm.save("model_close_svm.h5")
    # Make predictions and on Validation Data via individual models which would be input data of ensemble model

    pred_open_cnn =  model_open_cnn.predict(X_valid_cnn).flatten()
    pred_open_lstm = model_open_lstm.predict(X_valid_lstm).flatten() 
    pred_open_rf = model_open_rf.predict(X_valid).flatten()
    pred_open_svm = model_open_svm.predict(X_valid).flatten()

    pred_high_cnn = model_high_cnn.predict(X_valid_cnn).flatten()
    pred_high_lstm = model_high_lstm.predict(X_valid_lstm) .flatten()
    pred_high_rf = model_high_rf.predict(X_valid).flatten()
    pred_high_svm = model_high_svm.predict(X_valid).flatten()

    pred_low_cnn = model_low_cnn.predict(X_valid_cnn).flatten()
    pred_low_lstm = model_low_lstm.predict(X_valid_lstm).flatten()
    pred_low_rf = model_low_rf.predict(X_valid).flatten()  
    pred_low_svm = model_low_svm.predict(X_valid).flatten()

    pred_close_cnn = model_close_cnn.predict(X_valid_cnn).flatten()
    pred_close_lstm = model_close_lstm.predict(X_valid_lstm).flatten()
    pred_close_rf = model_close_rf.predict(X_valid).flatten()
    pred_close_svm = model_close_svm.predict(X_valid).flatten()

    # Train XGB model

    X_train_xgb_open = pd.DataFrame({'Open_cnn': pred_open_cnn, 'Open_lstm': pred_open_lstm, 'Open_rf': pred_open_rf, 'Open_svm': pred_open_svm})

    xgb_open = XGBRegressor()
    y_train_xgb_open = y_open_valid 
    xgb_open.fit(X_train_xgb_open, y_train_xgb_open)

    X_train_xgb_high = pd.DataFrame({'High_cnn': pred_high_cnn, 'High_lstm': pred_high_lstm, 'High_rf': pred_high_rf, 'High_svm': pred_high_svm})

    xgb_high = XGBRegressor()
    y_train_xgb_high = y_high_valid 
    xgb_high.fit(X_train_xgb_high, y_train_xgb_high)

    X_train_xgb_low = pd.DataFrame({'Low_cnn': pred_low_cnn, 'Low_lstm': pred_low_lstm, 'Low_rf': pred_low_rf, 'Low_svm': pred_low_svm})

    xgb_low = XGBRegressor()
    y_train_xgb_low = y_low_valid 
    xgb_low.fit(X_train_xgb_low, y_train_xgb_low)

    X_train_xgb_close = pd.DataFrame({'Close_cnn': pred_close_cnn, 'Close_lstm': pred_close_lstm, 'Close_rf': pred_close_rf, 'Close_svm': pred_close_svm})

    xgb_close = XGBRegressor() 
    y_train_xgb_close = y_close_valid 
    xgb_close.fit(X_train_xgb_close, y_train_xgb_close)

    xgb_open.save("xgb_open.h5")
    xgb_high.save("xgb_high.h5")
    xgb_low.save("xgb_low.h5")
    xgb_close.save("xgb_close.h5")


    # Make predictions on test set

    pred_open_cnn_test = model_open_cnn.predict(X_test_cnn).flatten()
    pred_open_lstm_test = model_open_lstm.predict(X_test_lstm).flatten()
    pred_open_rf_test = model_open_rf.predict(X_test).flatten()
    pred_open_svm_test = model_open_svm.predict(X_test).flatten()

    X_test_xgb_open = pd.DataFrame({'Open_cnn': pred_open_cnn_test, 'Open_lstm': pred_open_lstm_test, 'Open_rf': pred_open_rf_test, 'Open_svm': pred_open_svm_test})
    pred_open_xgb = xgb_open.predict(X_test_xgb_open)


    pred_high_cnn_test = model_high_cnn.predict(X_test_cnn).flatten()
    pred_high_lstm_test = model_high_lstm.predict(X_test_lstm).flatten()
    pred_high_rf_test = model_high_rf.predict(X_test).flatten()
    pred_high_svm_test = model_high_svm.predict(X_test).flatten()

    X_test_xgb_high = pd.DataFrame({'High_cnn': pred_high_cnn_test, 'High_lstm': pred_high_lstm_test, 'High_rf': pred_high_rf_test, 'High_svm': pred_high_svm_test})
    pred_high_xgb = xgb_high.predict(X_test_xgb_high)


    pred_low_cnn_test = model_low_cnn.predict(X_test_cnn).flatten()
    pred_low_lstm_test = model_low_lstm.predict(X_test_lstm).flatten()
    pred_low_rf_test = model_low_rf.predict(X_test).flatten()
    pred_low_svm_test = model_low_svm.predict(X_test).flatten()

    X_test_xgb_low = pd.DataFrame({'Low_cnn': pred_low_cnn_test, 'Low_lstm': pred_low_lstm_test, 'Low_rf': pred_low_rf_test, 'Low_svm': pred_low_svm_test})
    pred_low_xgb = xgb_low.predict(X_test_xgb_low)

    pred_close_cnn_test = model_close_cnn.predict(X_test_cnn).flatten()
    pred_close_lstm_test = model_close_lstm.predict(X_test_lstm).flatten()
    pred_close_rf_test = model_close_rf.predict(X_test).flatten()
    pred_close_svm_test = model_close_svm.predict(X_test).flatten()

    X_test_xgb_close = pd.DataFrame({'Close_cnn': pred_close_cnn_test, 'Close_lstm': pred_close_lstm_test, 'Close_rf': pred_close_rf_test, 'Close_svm': pred_close_svm_test})
    pred_close_xgb = xgb_close.predict(X_test_xgb_close)



    # Next day prediction
    # Make predictions for the 'Open' price
    pred_open_cnn_next_day = model_open_cnn.predict(X_test_cnn[-1:]).flatten()
    pred_open_lstm_next_day = model_open_lstm.predict(X_test_lstm[-1:]).flatten()
    pred_open_rf_next_day = model_open_rf.predict(X_test[-1:]).flatten()
    pred_open_svm_next_day = model_open_svm.predict(X_test[-1:]).flatten()

    # Create a DataFrame for the next day's data for the 'Open' price
    X_next_day_xgb_open = pd.DataFrame({'Open_cnn': pred_open_cnn_next_day, 'Open_lstm': pred_open_lstm_next_day, 'Open_rf': pred_open_rf_next_day, 'Open_svm': pred_open_svm_next_day})

    # Make predictions for the 'Open' price using the XGBoost model
    pred_open_xgb_next_day = xgb_open.predict(X_next_day_xgb_open)



    # Make predictions for the 'HIGH' price
    pred_high_cnn_next_day = model_high_cnn.predict(X_test_cnn[-1:]).flatten()
    pred_high_lstm_next_day = model_high_lstm.predict(X_test_lstm[-1:]).flatten()
    pred_high_rf_next_day = model_high_rf.predict(X_test[-1:]).flatten()
    pred_high_svm_next_day = model_high_svm.predict(X_test[-1:]).flatten()

    # Create a DataFrame for the next day's data for the 'high' price
    X_next_day_xgb_high = pd.DataFrame({'High_cnn': pred_high_cnn_next_day, 'High_lstm': pred_high_lstm_next_day, 'High_rf': pred_high_rf_next_day, 'High_svm': pred_high_svm_next_day})

    # Make predictions for the 'high' price using the XGBoost model
    pred_high_xgb_next_day = xgb_high.predict(X_next_day_xgb_high)

    # Make predictions for the 'low' price
    pred_low_cnn_next_day = model_low_cnn.predict(X_test_cnn[-1:]).flatten()
    pred_low_lstm_next_day = model_low_lstm.predict(X_test_lstm[-1:]).flatten()
    pred_low_rf_next_day = model_low_rf.predict(X_test[-1:]).flatten()
    pred_low_svm_next_day = model_low_svm.predict(X_test[-1:]).flatten()

    # Create a DataFrame for the next day's data for the 'Open' price
    X_next_day_xgb_low = pd.DataFrame({'Low_cnn': pred_low_cnn_next_day, 'Low_lstm': pred_low_lstm_next_day, 'Low_rf': pred_low_rf_next_day, 'Low_svm': pred_low_svm_next_day})

    # Make predictions for the 'Low' price using the XGBoost model
    pred_low_xgb_next_day = xgb_low.predict(X_next_day_xgb_low)


    # Make predictions for the 'Close' price
    pred_close_cnn_next_day = model_close_cnn.predict(X_test_cnn[-1:]).flatten()
    pred_close_lstm_next_day = model_close_lstm.predict(X_test_lstm[-1:]).flatten()
    pred_close_rf_next_day = model_close_rf.predict(X_test[-1:]).flatten()
    pred_close_svm_next_day = model_close_svm.predict(X_test[-1:]).flatten()

    # Create a DataFrame for the next day's data for the 'Close' price
    X_next_day_xgb_close = pd.DataFrame({'Close_cnn': pred_close_cnn_next_day, 'Close_lstm': pred_close_lstm_next_day, 'Close_rf': pred_close_rf_next_day, 'Close_svm': pred_close_svm_next_day})

    # Make predictions for the 'Close' price using the XGBoost model
    pred_close_xgb_next_day = xgb_close.predict(X_next_day_xgb_close)

    # Store the predictions for the next day's 'Close' price
    next_day_predictions = pd.DataFrame({'Open': pred_open_xgb_next_day, 'High':pred_high_xgb_next_day,'Low':pred_low_xgb_next_day,'Close':pred_close_xgb_next_day})

    print(next_day_predictions)

    # Evaluate metrics
    mse_open = mean_squared_error(y_open_test, pred_open_xgb)
    mse_high = mean_squared_error(y_high_test, pred_high_xgb)  
    mse_low = mean_squared_error(y_low_test, pred_low_xgb)
    mse_close = mean_squared_error(y_close_test, pred_close_xgb)

    mae_open = mean_absolute_error(y_open_test, pred_open_xgb) 
    mae_high = mean_absolute_error(y_high_test, pred_high_xgb)
    mae_low = mean_absolute_error(y_low_test, pred_low_xgb)
    mae_close = mean_absolute_error(y_close_test, pred_close_xgb)

    r2_open = r2_score(y_open_test, pred_open_xgb)
    r2_high = r2_score(y_high_test, pred_high_xgb)
    r2_low = r2_score(y_low_test, pred_low_xgb)
    r2_close = r2_score(y_close_test, pred_close_xgb)

    print(colored("The following results are for {}:".format(ticker), "magenta"))

    column1_name_length = max(len("Errors of Open Prices"), len("Errors of High Prices"), len("Errors of Low Prices"), len("Errors of Close Prices"), len("Errors of Volume"))

    print("{:<{column1_width}} {:>10} {:>10} {:>10}".format('', 'MAE', 'MSE', 'R^2', column1_width=column1_name_length))

    print("{:<{column1_width}} {:10.4f} {:10.4f} {:10.4f}".format('Errors of Open Prices', mae_open, mse_open, r2_open, column1_width=column1_name_length))

    print("{:<{column1_width}} {:10.4f} {:10.4f} {:10.4f}".format('Errors of High Prices', mae_high, mse_high, r2_high, column1_width=column1_name_length))

    print("{:<{column1_width}} {:10.4f} {:10.4f} {:10.4f}".format('Errors of Low Prices', mae_low, mse_low, r2_low, column1_width=column1_name_length))

    print("{:<{column1_width}} {:10.4f} {:10.4f} {:10.4f}".format('Errors of Close Prices', mae_close, mse_close, r2_close, column1_width=column1_name_length))

    # Display results
    results = pd.DataFrame(index=y_open_test.index)
    results['Actual Open'] = y_open_test
    results['Predicted Open'] = pred_open_xgb
    results['Actual High'] = y_high_test
    results['Predicted High'] = pred_high_xgb
    results['Actual Low'] = y_low_test  
    results['Predicted Low'] = pred_low_xgb
    results['Actual Close'] = y_close_test
    results['Predicted Close'] = pred_close_xgb
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(results)
    results.to_csv(f"Results_{ticker}.csv")

    #Forcast for next day
    # if not results.empty:
    next_day_data = pd.DataFrame(index=y_open_test.index)
    next_day_data['Open'] = results['Predicted Open'].iloc[-1]
    next_day_data['High'] = results['Predicted High'].iloc[-1]
    next_day_data['Low'] = results['Predicted Low'].iloc[-1]
    next_day_data['Close'] = results['Predicted Close'].iloc[-1]
    print(next_day_data)