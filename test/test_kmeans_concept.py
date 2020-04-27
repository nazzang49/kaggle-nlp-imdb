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
start = time.time()

# feature vecs of words
word_vectors = model.wv.syn0
# number of clusters -> 1 / 5 size of words
num_clusters = word_vectors.shape[0] / 5
num_clusters = int(num_clusters)

# define K-means and learning -> spend a bit of time
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# end - start -> time
end = time.time()
elapsed = end - start
print(elapsed, 'seconds')

idx = list(idx)
names = model.wv.index2word
# make centroid map
word_centroid_map = {names[i]: idx[i] for i in range(len(names))}

# print 10 words in each clusters (first clustering)
for cluster in range(0, 10):
    print('\nCluster {}'.format(cluster))

    words = []
    for i in range(0, len(list(word_centroid_map.values()))):
        if(list(word_centroid_map.values())[i] == cluster):
            words.append(list(word_centroid_map.keys())[i])
    print(words)