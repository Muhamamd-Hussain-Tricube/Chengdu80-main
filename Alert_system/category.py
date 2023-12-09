import requests
import json

# Set your API key and endpoint
subscription_key = "92abe7bf0a2e4951b0f31b087c069df4"
endpoint = "https://api.bing.microsoft.com/v7.0/news/trendingtopics"

# Set headers and parameters for the request
headers = {"Ocp-Apim-Subscription-Key": subscription_key}
params = {"mkt": "en-US"}

# Send the request and get the response
response = requests.get(endpoint, headers=headers, params=params)
data = json.loads(response.text)

# print(data)
# Log the trending topics
for topic in data['value']:
    print(topic["name"], topic[])
