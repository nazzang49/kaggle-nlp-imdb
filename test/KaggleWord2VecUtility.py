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
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# steps -> review_to_sentences (1) - review_to_words (2)
class KaggleWord2VecUtility(object):

    # stopwords flag default = False
    def review_to_words(raw_review, remove_stopwords=False):
        # step1
        text_only = BeautifulSoup(raw_review, 'html.parser').getText()
        # step2
        letters_only = re.sub('[^a-zA-Z]', ' ', text_only)
        # step3
        words = letters_only.lower().split()
        # step4
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        # step5
        stemmer = SnowballStemmer('english')
        stemming_words = [stemmer.stem(w) for w in words]
        # return list type
        return (stemming_words)

    def review_to_join_words(review, remove_stopwords=False):
        words = KaggleWord2VecUtility.review_to_words(review, remove_stopwords=False)
        join_words = ' '.join(words)
        return join_words

    def review_to_sentences(review, remove_stopwords=False):
        # load the punkt tokenizer
        """
        pickle -> persist the reference of all variables
        """
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # 1. nltk tokenizer -> tokenization + remove all blanks
        raw_sentences = tokenizer.tokenize(review.strip())
        # 2. circle each sentences
        sentences = []
        for raw_sentence in raw_sentences:
            # if empty -> skip and done
            if len(raw_sentence) > 0:
                # preprocessing of each sentences
                sentences.append(KaggleWord2VecUtility.review_to_words(raw_sentence, remove_stopwords))
        # return preprocessed dataset by array type
        return sentences