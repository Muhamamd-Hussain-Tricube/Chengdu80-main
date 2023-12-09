import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import pickle
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, SimpleRNN, Bidirectional
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import tensorflow as tf
seed = 42
# tf.__version__

import warnings 
warnings.filterwarnings("ignore")

tf.random.set_seed(seed)

import os
print("Dataset Path: ")
for dirname, _, filenames in os.walk('/Users/mubeen/Documents/Chengdu_Real/Chengdu80/Fraud_Detection/fake-news'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

class Model:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.sent_length = None
        self.voc = 10000
        self.messages = None
        self.X_final = None
        self.y_final = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.test_loss = None
        self.test_acc = None

    def import_libraries(self):
        pass
        # import seaborn as sns
        # import nltk
        # import re
        # from nltk.corpus import stopwords
        # nltk.download('stopwords')
        # from nltk.stem.porter import PorterStemmer
        # import pickle
        # from tensorflow.keras.layers import Embedding
        # from tensorflow.keras.preprocessing.sequence import pad_sequences
        # from tensorflow.keras.models import Sequential
        # from tensorflow.keras.preprocessing.text import one_hot
        # from tensorflow.keras.layers import Dropout
        # from tensorflow.keras.layers import LSTM, SimpleRNN, Bidirectional
        # from tensorflow.keras.layers import Dense
        # from sklearn.model_selection import train_test_split
        # from sklearn.metrics import confusion_matrix
        # from sklearn.metrics import accuracy_score
        # from sklearn.metrics import classification_report

    def load_data(self):
        self.train = pd.read_csv(self.train)
        self.test = pd.read_csv(self.test)



    def preprocess_data(self):
        # Preprocess data
        df = self.train.dropna()
        X = df.drop("label", axis = 1)
        y = df["label"]

        self.messages=X.copy()
        self.messages['title'][1]
        self.messages.reset_index(inplace=True)

        # Vectorization 
        ps = PorterStemmer()
        corpus = []
        for i in range(0, len(self.messages)):
            review = re.sub('[^a-zA-Z]', ' ', self.messages['title'][i])
            review = review.lower()
            review = review.split()
            
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)

        # one hot representation
       
        onehot_repr=[one_hot(words,self.voc)for words in corpus]

        self.sent_length = 20 
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=self.sent_length)
        # embedded_docs, len(embedded_docs),y.shape

        self.X_final=np.array(embedded_docs)
        self.y_final=np.array(y)




    def split_data(self):
        # Split data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_final, self.y_final, test_size=0.33, random_state=42)


    def define_model(self):
        # Define Bi-LSTM model
        embedding_vector_features=40
        self.model = Sequential()
        self.model.add(Embedding(self.voc,embedding_vector_features,input_length=self.sent_length))
        self.model.add(Bidirectional(LSTM(100)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1,activation='sigmoid'))
     


    def compile_model(self):
        # Compile model
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        print(self.model.summary())

    def train_model(self, epochs, batach_size):
        # Train model
        self.model.fit(self.x_train, self.y_train,validation_data=(self.x_test,self.y_test),epochs=epochs,batch_size=batach_size)


    def predict(self):
        # Make predictions 
        y_pred=self.model.predict(self.x_test)
        y_pred=(y_pred>0.5).astype(int)
        print(confusion_matrix(self.y_test,y_pred))
        print(accuracy_score(self.y_test,y_pred))
        print(classification_report(self.y_test,y_pred))

    def new_predict(x_test):
    # Make predictions 
        y_pred=self.model.predict(x_test)
        y_pred=(y_pred>0.5).astype(int)
        print(y_pred)


    def new_data_predict(self, new_data):
        # Make predictions on new data
        self.x_test = new_data
        self.preprocess_data()
        predictions = self.model.predict(new_data)
        return predictions

    def save_model(self):
        # Save the trained model
        self.model.save("saved_model")
        # Save as pickle file
        pickle.dump(self.model, open("model.pkl", "wb"))
        

    # def save_tokenizer(self):
    #     # Save the tokenizer
    #     with open("tokenizer.pickle", 'wb') as handle:
    #         pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    model = Model("/Users/mubeen/Documents/Chengdu_Real/Chengdu80/Fraud_Detection/fake-news/train.csv", "/Users/mubeen/Documents/Chengdu_Real/Chengdu80/Fraud_Detection/fake-news/test.csv")
    model.import_libraries()
    model.load_data()
    model.preprocess_data()
    model.split_data()
    model.define_model()
    model.compile_model()
    model.train_model(epochs=10, batach_size=64)
    predictions = model.predict()

    model.save_model()
    # model.save_tokenizer() 




    # predictions = model.new_data_predict(new_data)


# if __name__ == '__main__':
#     main()



if __name__ == '__main__':
    print(main())



