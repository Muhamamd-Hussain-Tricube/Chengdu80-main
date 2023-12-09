from transformers import pipeline


# Combined Data format
# Combined_data = {'2021-03-01': ['description1','description2','description3'], '2021-03-02': ['description1','description2','description3']}


def get_sentiment_score(combined_data):

    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert",padding=True) # device -1 for 

    # Get Sentiment Score for each date in combined_data
    sentiment_score = {}
    for date, descriptions in combined_data.items():
        sentiment_score[date] = pipe(descriptions)

    sentiment_score

    # Make Score for negative sentiment to be negative

    for date, score in sentiment_score.items():
        for s in score:
            if s['label'] == 'negative':
                s['score'] = -s['score']

    sentiment_score

    # Take AVG of sentiment score per date
    avg_sentiment_score = {}
    for date, score in sentiment_score.items():
        avg_sentiment_score[date] = sum([s['score'] for s in score])/len(score)

    avg_sentiment_score



