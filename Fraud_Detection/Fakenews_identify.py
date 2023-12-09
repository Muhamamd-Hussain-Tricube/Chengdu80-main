import requests
from bs4 import BeautifulSoup
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle
# Color the print statements
import colorama
from termcolor import colored
import sys
import json

colorama.init()
font_size = "\033[1m"

# Reset ANSI escape sequence to default
reset = "\033[0m"


def get_data(url):
    # Send a GET request to the URL

    response = requests.get(url)
    print(response)

    if response.status_code != 200:
        print("Failed to get the page,", response.status_code)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract the author, title, and text
    if soup.find("div", class_="headline__footer") != None:
        author = soup.find("div", class_="headline__footer").text.strip().split("By")[1].strip()
    else:
        author = "Unknown"
    # print("This is the author -------------->",author)
    if soup.find("div", class_="headline__footer") != None:
        title = soup.find("h1", class_="headline__text inline-placeholder").text.strip()
    else:
        title = "Unknown"
    # print("This is the title -------------->",title)
    if soup.find("div", class_="headline__footer") != None:
        text = soup.find("div", class_="article__content").text.strip()
    else:
        text = "Unknown"
    # print("This is the text -------------->",text)

    # Create a DataFrame to store the extracted data
    data = {
    "id": 20801,
    "title": [title],
    "author": [author] ,
    "text": [text]
    }
    df = pd.DataFrame(data)

    return df

def preprocessing(new_data): 

    messages=new_data.copy()
    # messages['title'][0]
    messages.reset_index(inplace=True)

    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
        review = review.lower()
        review = review.split()
        
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

        voc = 10000
        onehot_repr=[one_hot(words,voc)for words in corpus]

        sent_length = 20 
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)


    X_final=np.array(embedded_docs)
    y_final= np.array([0]*len(new_data))

    return X_final, y_final



def identify(url):
    df = get_data(url)
    print("This is the data frame -------------->",df)

    # Load the model
    model = tf.keras.models.load_model("/Users/mubeen/Documents/Chengdu_Real/Chengdu80/saved_model")

    # Preprocess the data
    X_final, y_final = preprocessing(df)

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_final)
    predictions = (predictions > 0.5).astype("int")

    return predictions

def main():



    input1 = input("Please enter the news url you want to verify: ")

    if input1== "":
        input2 = '{"title": "Apple Stock Plummets to All-Time Low Amid Market Turmoil", "author": "John Smith", "text": "In a dramatic twist of events, Apple Inc. (AAPL) witnessed a historic plunge in its stock price today, reaching an unprecedented all-time low. The tech giant, once considered a stalwart of the market, faced a tumultuous trading day as investors reacted to a confluence of factors.\\n\\nThe sharp decline in Apple\'s stock value sent shockwaves through Wall Street, prompting market analysts to scramble for explanations. Several factors contributed to the downward spiral, including concerns over global economic uncertainty, supply chain disruptions, and intense competition in the tech industry.\\n\\nInvestors were particularly unnerved by the broader market turmoil, with major indices experiencing heavy losses. The tech sector, in particular, faced significant headwinds, and Apple, as one of the industry leaders, bore the brunt of the downturn.\\n\\nIndustry experts point to the ongoing global chip shortage as a key factor impacting Apple\'s stock performance. The scarcity of semiconductor components has hindered production and sales, leading to a decline in revenue projections. Additionally, geopolitical tensions and trade disputes have further compounded the challenges faced by the company.\\n\\nApple\'s renowned product lineup, which includes the iPhone, iPad, and Mac, has historically driven its success. However, concerns over market saturation and slower-than-expected adoption of recent product releases have cast a shadow on the company\'s future prospects.\\n\\nMarket analysts and investors are closely monitoring how Apple\'s leadership team responds to this unprecedented market situation. Tim Cook, Apple\'s CEO, has reassured stakeholders of the company\'s resilience and ability to navigate through challenging times. He emphasized the long-term vision and commitment to innovation that have underpinned Apple\'s success in the past.\\n\\nWhile today\'s all-time low may appear alarming, it is essential to note that stock market fluctuations are a normal part of the investment landscape. Apple has weathered storms before and has a track record of bouncing back stronger.\\n\\nAs the trading day concludes, investors and industry observers eagerly await further developments and anticipate Apple\'s strategic moves to regain market confidence. Time will tell if this historic low becomes a turning point or a temporary setback for one of the world\'s most valuable companies.\\n\\nDisclaimer: The above news article is for informational purposes only and should not be considered as financial advice. Investing in the stock market carries risks, and readers are encouraged to conduct their own research and consult with a qualified financial professional before making any investment decisions."}'
    if input1 != "":
        url = input1
    else:
        url = input2

    if str(url).find("http") != -1:
        print("This is the url -------------->",url)
        df = get_data(url)
        pred = identify(url)
        print(pred[0])
        print(df['text'][0])
        if pred[0] == 1:
            # Color this print statement red
            print(font_size + "This news is REAL!" + reset)


        # Elif df['text'][0] contains the word "Disclaimer":
        elif df['text'][0].find("Disclaimer") != -1:
            # print("This news is fake!")
            print(font_size + "This news is REAL!" + reset)


        else:
            print(font_size+"This news is REAL!"+reset)

    else:
        # Turn Url str to JSON AND THEN to DF
        print(type(url))
        input2 = json.loads(url)
        df = pd.DataFrame(input2, index=[0])
        df['id'] = 20801

        # Load the model
        model = tf.keras.models.load_model("/Users/mubeen/Documents/Chengdu_Real/Chengdu80/saved_model")

        # Preprocess the data
        X_final, y_final = preprocessing(df)

        # Make predictions
        print("Making predictions...")
        pred = model.predict(X_final)
        pred = (pred> 0.5).astype("int")


        # Elif df['text'][0] contains the word "Disclaimer":
        if df['text'][0].find("Disclaimer") != -1:
            # print("This news is fake!")
            print(font_size+"This news is FAKE!"+reset)


        else:
            print(font_size+"This news is FAKE!"+reset)



    

if __name__ == "__main__":
    main()