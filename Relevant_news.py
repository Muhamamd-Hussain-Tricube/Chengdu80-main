import spacy
import pandas as pd
import json
import os
from pprint import pprint
import requests

nlp = spacy.load('en_core_web_sm')

def get_stock(stocklist, txt_file):

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(stocklist)

        # Combine the 'Symbol' and 'Name' columns into a single list and convert to lower case
        stocks = df['Symbol'].tolist() + df['Name'].tolist()
        stocks = [stock.lower() for stock in stocks]

        # Load the English model for spaCy
        

        # Read the article text file
        with open(txt_file, 'r') as file:
            article = file.read().replace('\n', '')

        # Process the article with spaCy
        doc = nlp(article)

        # Extract the entities from the article
        entities = [ent.text.lower() for ent in doc.ents]
        # print(entities)
        # Initialize an empty list to store the mentioned stocks
        mentioned_stocks = []

        # Check each stock to see if it is mentioned in the entities
        for stock in stocks:
            if stock in entities:
                mentioned_stocks.append(stock)

        return mentioned_stocks

        

relevent_stocks = get_stock("constituent stocks","text to be read")



subscription_key = "Api key"
search_term = relevent_stocks
search_url = "https://api.bing.microsoft.com/v7.0/news/search"

def news_per_company(company):
    headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
    params  = {"q": company, "textDecorations":True, "textFormat":"HTML","mkt": "en-US"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    print(type(search_results))
    return search_results

def create_data(search_term):
    data = []
    for i in search_term:
        data.append(news_per_company(i))
    return data
    
data = create_data(search_term)

print(data)

