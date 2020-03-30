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
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
# multiprocessing
from multiprocessing import Pool
# wordcloud and graph packages
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# show graphs
import seaborn as sns

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

# lemmatizing -> infer real meaning of word based on contexts back and forth e.g) different meaning between norm and verb
wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize('fly'))
print(wordnet_lemmatizer.lemmatize('flies'))
words = [wordnet_lemmatizer.lemmatize(w) for w in words]

# make a def including whole process of upper steps
def review_to_words(raw_review):
    # step1
    text_only = BeautifulSoup(raw_review, 'html.parser').getText()
    # step2
    letters_only = re.sub('[^a-zA-Z]', ' ', text_only)
    # step3
    words = letters_only.lower().split()
    # step4
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
    # step5
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # return final sentence combined by blank
    return(' '.join(stemming_words))

# check def
print(review_to_words(train['review'][0]))

# below 2-def are processes for reducing preprocessing time
def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

# df -> dataframe = datasets
# func -> specific function
# **kwargs -> arguments defined by user (can be various -> pop by key)
def apply_by_multiprocessing(df, func, **kwargs):
    # get workers parameter from keywords
    workers = kwargs.pop('workers')
    # define process pool by workers
    pool = Pool(processes=workers)
    # multiprocessing
    result = pool.map(_apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    # combine each results
    return pd.concat(list(result))

# preprocessed_dataset = apply_by_multiprocessing(train['review'], review_to_words, workers=4)

preprocessed_dataset_train = train['review'].apply(review_to_words)
preprocessed_dataset_test = train['review'].apply(review_to_words)

# import wordcloud (not necessary)
def display_wordcloud(data=None, backgroundcolor='white', width=800, height=600):
    wordcloud=WordCloud(stopwords=STOPWORDS, background_color=backgroundcolor, width=width, height=height).generate(data)
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

# display_wordcloud(' '.join(preprocessed_dataset_train))

# total words count
train['num_words'] = preprocessed_dataset_train.apply(lambda x: len(str(x).split()))
# total words count without duplication
train['num_uniq_words'] = preprocessed_dataset_test.apply(lambda x: len(set(str(x).split())))

# check (1st review)
x = preprocessed_dataset_train[0]
x = str(x).split()
print(len(x))
x[:10]

# multiple graphs
fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(18, 6)
print('리뷰별 단어 평균 값 :', train['num_words'].mean())
print('리뷰별 단어 중간 값', train['num_words'].median())
sns.distplot(train['num_words'], bins=100, ax=axes[0])
axes[0].axvline(train['num_words'].median(), linestyle='dashed')
axes[0].set_title('리뷰별 단어 수 분포')

print('리뷰별 고유 단어 평균 값 :', train['num_uniq_words'].mean())
print('리뷰별 고유 단어 중간 값', train['num_uniq_words'].median())
sns.distplot(train['num_uniq_words'], bins=100, color='g', ax=axes[1])
axes[1].axvline(train['num_uniq_words'].median(), linestyle='dashed')
axes[1].set_title('리뷰별 고유한 단어 수 분포')

