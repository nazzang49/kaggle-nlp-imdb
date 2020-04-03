# multi-classification based on fail log return message
# (remember) prevent this file from uploading to private github
import pandas as pd
from konlpy.tag import Okt
import json
import os
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import numpy as np
import re
# tagging korean sentence by Okt class
from sklearn.pipeline import Pipeline

okt = Okt()

# before reading this file -> replace (,) to blank / replace tab to blank
# quoting=3 means ignoring double-quota (")
fail_log = pd.read_csv('D:/fail_log/ts_pg_fail_log_03.csv', header=0, delimiter=',', quoting=3, encoding='euc-kr')
# shuffling
fail_log = fail_log.sample(frac=1)

# divide into X, Y
fail_log_x = fail_log['pfl_return_message']
fail_log_y = fail_log['pfl_issue_type']
# divide into train, test (random_state is a seed number for shuffling)
X_train, X_test, Y_train, Y_test = train_test_split(fail_log_x, fail_log_y, test_size=0.3, random_state=123)

import nltk
# use download command cuz of nltk bug
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# data preprocessing
# make a def including whole process of upper steps
def review_to_words(raw_review):
    # subtract only korean characters
    letters_only = re.sub('[^가-힣]+', ' ', raw_review)
    # norm -> regularization
    # stem -> stemming
    temp = [t for t in okt.morphs(letters_only, stem=True, norm=True)]
    return ' '.join(temp)

train_dataset = X_train.apply(review_to_words)
test_dataset = X_test.apply(review_to_words)

# vectorize the train_dataset
vectorizer = CountVectorizer()
vectorizer.fit(train_dataset)

train_data_features = vectorizer.transform(train_dataset)
print(train_data_features.shape)
# vectorized features (train_X)
train_X = train_data_features.toarray()

test_data_features = vectorizer.transform(test_dataset)
print(test_data_features.shape)
# vectorized features (test_X)
test_X = train_data_features.toarray()

from tensorflow.keras.utils import to_categorical
# replace Y dataset to one-hot-encoding vector
train_Y = to_categorical(Y_train)
test_Y = to_categorical(Y_test)

print(train_Y.shape)
print(test_Y.shape)

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_dim=344))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])

# epoch = total loop count
# batch_size = learning size at 1 epoch
model.fit(train_X, train_Y, epochs=10, batch_size=512, use_multiprocessing=True)
results = model.evaluate(test_X, test_Y)

# saving this model, and load when you need
from keras.models import load_model
model.save('fail_log_automization_model.h5')