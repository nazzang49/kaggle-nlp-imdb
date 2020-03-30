# kaggle bag-of-words NLP project test with inflearn video clips
# dataset from IMDb website
import pandas as pd
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
# packages for regex
import re
import nltk
# use download command cuz of nltk bug
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# train dataset -> delimiter is tab / 25000 rows
train = pd.read_csv("C:/bag-of-words-dataset/labeledTrainData.tsv",
                    header=0, delimiter='\t', quoting=3)

# test dataset -> delimiter is tab / 25000 rows
test = pd.read_csv("C:/bag-of-words-dataset/testData.tsv",
                    header=0, delimiter='\t', quoting=3)

# check if the dataset is called without any errors
# shape = dimension of array
print(train.head())
print(train.shape)
print(test.head())
print(test.shape)

# check names of each columns
print(train.columns.values)
print(test.columns.values)

# dataset preprocessing -> 4 steps below
# get rid of Stopwords with NLTK packages e.g) i, my, me, mine
# learn Stemming and Lemmatizing, do Stemming with SnowballStemmer

# (step1) get rid of html tags with BeautifulSoup which is used when crawling
ex1 = BeautifulSoup(train['review'][0], 'html5lib')
# print(train['review'][0][:700])
print(ex1.getText()[:700])

# (step2) replace languages, special signs to blanks with regex except the english
# ', !, ? -> blank
words_only = re.sub('[^a-zA-Z]', ' ', ex1.getText())
print(words_only[:700])
# toLowerCase and split sentences by blank (tokenization)
words_lower_case = words_only.lower()
words = words_lower_case.split()
print(len(words))

# (step3) get rid of Stopwords with NLTK packages e.g) i, my, me, mine, this, that, is, are
# NLTK defines specific Stopwords in their packages not including Korean
# check which words are defined as Stopwords
print(stopwords.words('english')[:10])
words = [w for w in words if not w in stopwords.words('english')]
# 437 -> 219
print(len(words))

# (step4) learn Stemming and Lemmatizing, do Stemming with SnowballStemmer
# Stemming -> categorizing similar words into same meaningful one word (형태소 분석)
    # porter = passive
    # lancaster = progressive

stemmer = SnowballStemmer('english')
words = [stemmer.stem(w) for w in words]
print(words[:10])

