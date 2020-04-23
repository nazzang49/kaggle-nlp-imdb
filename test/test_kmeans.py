# K-means clustering algorithm -> grouping vector by relative rate (vector quantization = 벡터 양자화) / find out cluster word which is a center

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
import time
from nltk.corpus import stopwords
import nltk.data
import matplotlib.pyplot as plt
import seaborn as sns

# load Word2Vec model
model = Word2Vec.load('300features_40minwords_10text')

# numpy -> ndarray
print(type(model.wv.syn0))
# 11986, 300 -> all words expressed by 300-dim vectors
print(model.wv.syn0.shape)
# check
print(model.wv['flower'][:10])

# what is K-means clustering -> https://ko.wikipedia.org/wiki/K-%ED%8F%89%EA%B7%A0_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
    # grouping by relative feature
        # scale revision
            # (step1) pick random vectors based on centroid
            # (step2) assign these samples to the nearest centroid
            # (step3) re-calculate the position of centroid
            # repeat step2 and step3 until centroid doesn't move

# actual test
