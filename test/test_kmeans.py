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

# read data as DataFrame by pandas
# data preprocessing like previous tutorial -> clean review (train, test)
train = pd.read_csv('C:/bag-of-words-dataset/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
test = pd.read_csv('C:/bag-of-words-dataset/testData.tsv', header=0, delimiter='\t', quoting=3)

from KaggleWord2VecUtility import KaggleWord2VecUtility

# refinement of train, test dataset
clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(KaggleWord2VecUtility.review_to_words(review, remove_stopwords=True))

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append(KaggleWord2VecUtility.review_to_words(review, remove_stopwords=True))

# make bag of centroids
train_centroids = np.zeros((train['review'].size, num_clusters), dtype='float32')
# check
print(train_centroids[:5])

# centroid -> distance between two clusters
def create_bag_of_centroids(wordlist, word_centroid_map):
    # word_centroid_map.values() -> idx = cluster
    # word_centroid_map.keys() -> words
    num_centroids = max(word_centroid_map.values()) + 1

    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            # how many words included in specific cluster
            bag_of_centroids[index] += 1

    # return bag of centroids
    return bag_of_centroids

# calculate count of words of each sentences by bag of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

forest = RandomForestClassifier(n_estimators=100)

print("========================learning train dataset========================")
forest.fit(train_centroids, train["sentiment"])

from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(forest, train_centroids, train['sentiment'], cv=10, scoring='roc_auc'))
result = forest.predict(test_centroids)

print(score)
print(result)

# exporting the result
output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
output.to_csv('c:/bag-of-words-dataset/submit_bag_of_centroids_{0:.5f}'.format(score), index=False, quoting=3)

# check by graph
fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.countplot(train['sentiment'], ax=axes[0])
sns.countplot(output['sentiment'], ax=axes[1])

