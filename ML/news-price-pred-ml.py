from sklearn.feature_extraction.text import CountVectorizer

import yfinance as yf
import pandas as pd
import numpy as np
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

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()

def get_document_embedding(row):
    document_embedding = torch.zeros(model.config.n_embd)
    for keyword, frequency in json.loads(row["keyword_frequency"]).items():
        # print(keyword)
        inputs = tokenizer(keyword, return_tensors='pt')
        outputs = model(**inputs)
        keyword_embedding = outputs.last_hidden_state.mean(dim=1)
        keyword_embedding = keyword_embedding.squeeze(0)

        document_embedding += frequency * keyword_embedding
    return document_embedding.detach().numpy()

def get_count_vector(text):
    vectorizer = CountVectorizer()
    count_vector = vectorizer.fit_transform([text])
    return count_vector.toarray().squeeze()


print("Reading CSV...")
# mega_csv = pd.read_csv('mega.csv')
# print(mega_csv.head())
# unique_tickers = mega_csv['ticker'].unique()
# print(mega_csv['ticker'])
# all_dfs = {ticker: mega_csv[mega_csv['ticker'] == ticker] for ticker in unique_tickers}

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
print(mega_csv.head())
unique_tickers = mega_csv['ticker'].unique()
all_dfs = {ticker: mega_csv[mega_csv['ticker'] == ticker] for ticker in unique_tickers}

tickers = unique_tickers
for ticker in tickers:

    print(f"Processing {ticker}...")
    df = all_dfs[ticker]
    print("Preparing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    print("Embedding headlines...")
    headline_embeddings = df['headline'].apply(get_count_vector)
    print(headline_embeddings)
    print("Embedding keywords...")
    document_embedding = df['keyword_frequency'].apply(get_count_vector)
    print(document_embedding)

    # print("Reducing headline dimensions...")
    # pca_headline = PCA(n_components=50)
    # reduced_headline_embeddings = pca_headline.fit_transform(headline_embeddings)

    # print("Reducing keywords dimensions...")
    # pca_keywords = PCA(n_components=50)
    # reduced_document_embeddings = pca_keywords.fit_transform(document_embedding)

    print("Downloading OHLC data...")
    ohlc_data = yf.download(ticker, start=df['Date'].min(), end=df['Date'].max())
    ohlc_data.reset_index(inplace=True)
    ohlc_data['Date'] = ohlc_data['Date'].dt.strftime('%Y-%m-%d')

    df = pd.merge(df, ohlc_data, on='Date', how='inner')

    df.fillna(method='ffill')

    features = pd.concat([pd.DataFrame(headline_embeddings, index=df.index), pd.DataFrame(document_embedding, index=df.index), df['sentiment_score']], axis=1)

    targets = df[['Open', 'High', 'Low', 'Close']]
    print(features)

    X_train, X_temp, y_train, y_temp = train_test_split(features, targets, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(X_train.isnull().sum())  # check for NaN values

    ############## CNN ################
    print("Training CNN Model...")
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(50, activation='relu'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Dense(4))

    cnn_model.compile(optimizer='adam', loss='mean_squared_error')
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    cnn_model.evaluate(X_test, y_test)
    cnn_predictions = cnn_model.predict(X_val)
    print(cnn_predictions)

    ############ LSTM ############
    print("Training LSTM Model...")
    X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
    X_val_lstm = np.reshape(X_val.values, (X_val.shape[0], X_val.shape[1], 1))
    X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(4))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)
    lstm_model.evaluate(X_test_lstm, y_test)
    lstm_predictions = lstm_model.predict(X_val_lstm)
    print(lstm_predictions)

    ########### RF #############
    print("Training RF Model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_val)
    print(rf_predictions)

    ########## XGB ###########
    print("Training XGB Ensemble Model...")
    ensemble_features = np.concatenate([cnn_predictions, lstm_predictions, rf_predictions], axis=1)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    xgb_model.fit(ensemble_features, y_val)

    cnn_test_predictions = cnn_model.predict(X_test)
    lstm_test_predictions = lstm_model.predict(np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1)))
    rf_test_predictions = rf_model.predict(X_test)

    ensemble_test_features = np.concatenate([cnn_test_predictions, lstm_test_predictions, rf_test_predictions], axis=1)

    xgb_test_predictions = xgb_model.predict(ensemble_test_features)
    print(xgb_test_predictions)

    xgb_test_predictions_df = pd.DataFrame(
        xgb_test_predictions, 
        columns=['Predicted Open', 'Predicted High', 'Predicted Low', 'Predicted Close'], 
        index = X_test.index
    )
    print(xgb_test_predictions_df)
    
    ########## EVALUATION ############
    print("Evaluating Metrics...")
    actual = y_test
    predicted = pd.DataFrame(xgb_test_predictions, columns=['Open', 'High', 'Low', 'Close'], index=y_test.index)
    metrics = pd.DataFrame(index=['Open', 'High', 'Low', 'Close'], columns=['MSE', 'MAE'])
    for column in ['Open', 'High', 'Low', 'Close']:
        mse = mean_squared_error(actual[column], predicted[column])
        mae = mean_absolute_error(actual[column], predicted[column])
        metrics.loc[column, 'MSE'] = mse
        metrics.loc[column, 'MAE'] = mae
    print(metrics)

    results = pd.concat([actual, predicted, metrics], keys=['Actual Values', 'Predicted Values', 'Metrics'])
    results.to_csv(f'{ticker}_ensemble_model_results.csv')