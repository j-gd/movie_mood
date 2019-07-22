import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from sklearn.svm import SVC
from nltk.stem.snowball import SnowballStemmer

import tensorflow as tf

from tensorflow import tensorflow.keras as keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import Dropout
from keras.layers import Bidirectional

np.random.seed(9)

MAX_NB_WORDS = 500
max_review_length = 500
EMBEDDING_DIM = 160


class LSTMModel():
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                      lower=True,
                      split=' ',
                      char_level=False,
                      oov_token=None,
                      document_count=0)

    def fit(self, text_list):
        tokenizer.fit_on_texts(text_list)
        print('Length of tokenizer word index:', len(tokenizer.word_index))

        nb_words  = min(MAX_NB_WORDS, len(word_index))
        lstm_out = max_review_length

        model = Sequential()
        model.add(Embedding(nb_words,EMBEDDING_DIM,input_length=max_review_length))
        #model.add(Dropout(0.2))

        ## add conv using kernal No.32 and size 3x3, actiation='relu'(rm neg)
        # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        # model.add(MaxPool1D(pool_size=2))
        model.add(Bidirectional(LSTM(40, return_sequences=True)))
        #model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(40)))
        #model.add(Bidirectional(LSTM(20)))
        #model.add(Attention(max_review_length))
        model.add(Dense(3, activation = 'softmax'))

        ## one-code mutiple categories targets use 'categorical_crossentropy' not 'binary_crossentropy'
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics =['accuracy'])
            
    
    def transform(self, text_list):
        train_sequences = tokenizer.texts_to_sequences(text_list)


