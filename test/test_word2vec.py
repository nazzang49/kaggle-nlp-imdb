# one hot encoding is not enough for general deep learning model cuz of memories and efficiency of NN
# instead, usually use Word2Vec concept for deep learning
# Word2Vec is known as focusing on mutual relation among words (=context)

import pandas as pd

# call review dataset
train = pd.read_csv('C:/bag-of-words-dataset/labeledTrainData.tsv',
                    header=0, delimiter='\t', quoting=3)

test = pd.read_csv('C:/bag-of-words-dataset/labeledTrainData.tsv',
                   header=0, delimiter='\t', quoting=3)

unlabeled_train = pd.read_csv('C:/bag-of-words-dataset/unlabeledTrainData.tsv',
                              header=0, delimiter='\t', quoting=3)


print(train.head())
print(test.head())

# make utility file for data preprocessing
# from -> call module (xxxx.py) import class name
from KaggleWord2VecUtility import KaggleWord2VecUtility

print(KaggleWord2VecUtility.review_to_words(train['review'][0])[:10])

sentences = []
for review in train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)

for review in unlabeled_train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)

# make log info
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# set parameters
num_features = 300  # dimension
min_word_count = 40 # ignore non-meaningful words which are not repeated several times in every reviews
num_workers = 4     # multiprocessing thread
context = 10        # at least needed words to analyze context
downsampling = 1e-3 # words' count (usually, 0.001)

from gensim.models import word2vec
# studying relation among words
model = word2vec.Word2Vec(sentences,
                          workers=num_workers,
                          size=num_features,
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)

# unload unnecessary memory
model.init_sims(replace=True)
model_name = '300features_40minwords_10text'
model.save(model_name)

# extract non-relative word -> kitchen
model.wv.doesnt_match('man woman child kitchen'.split())

# extract most-relative word -> lots of result
model.wv.most_similar('man')

# visualizing by t-SNE
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim
import gensim.models as g

# solve fraction problem of minus font
mpl.rcParams['axes.unicode_minus'] = False

# load saved model
model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X))
print(X[0][:10])
tsne = TSNE(n_components=2)

# visualizing for only 100 words (Actually, too many)
X_tsne = tsne.fit_transform(X[:100, :])

df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
df.shape

# wordcloud focusing on relation group
fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)

plt.show()