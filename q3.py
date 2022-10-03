import pandas as pd
import sklearn.manifold
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import re
import random
from tqdm import tqdm
import itertools
import textract



# Get the stopwords from the NLTK toolkit



def read_files():
    # Read all the Game of Thrones books and combine them into single corpus.
    data = textract.process('/home/zzy/PycharmProjects/Word2Vec-Using-Gensim/CV-ZiyuZhao-1101.docx')


    return data


def clean_data(sentence):
    # Clean the data from all punctuation and remove all the stopwords.
    sentence = re.sub(r'[^A-Za-z0-9\s.]', r'', str(sentence).lower())
    sentence = re.sub(r'\n', r' ', sentence)
    #sentence = " ".join([word for word in sentence.split() if word not in stopWords])

    return sentence


def shuffle_corpus(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)

    return shuffled


# Function to visualize the bag of words
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
        ]

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

data = read_files()
#%%
# Cleansing the big corpus and converting into DataFrame
data = data.splitlines()
data = list(filter(None, data))
data = pd.DataFrame(data)
#%%
# Do further cleaning and convert each sentences into tokens
data[0] = data[0].map(lambda x: clean_data(x))
tmp_corpus = data[0].map(lambda x: x.split('.'))

corpus = []
for i in tqdm(range(len(tmp_corpus))):
    for line in tmp_corpus[i]:
        words = [x for x in line.split()]
        corpus.append(words)

#print(corpus[0]) # Example of how each list in corpus looks like

num_of_sentences = len(corpus)
num_of_words = 0
for line in corpus:
    num_of_words += len(line)

#print('Num of sentences - %s'%(num_of_sentences))
#print('Num of words - %s'%(num_of_words))

# To detect the common phrases and combine them into a single word
phrases = Phrases(sentences=corpus,min_count=25,threshold=50)
bigram = Phraser(phrases)

for index,sentence in enumerate(corpus):
    corpus[index] = bigram[sentence]

# Define the required parameters
size = 100
window_size = 2 # sentences weren't too long
epochs = 100
min_count = 2
workers = 4

model = Word2Vec(corpus, sg=1, window=window_size, vector_size=size, min_count=min_count, workers=workers, epochs=epochs, sample=0.01)

model.build_vocab(shuffle_corpus(corpus), update=True)

# Training the model
for i in tqdm(range(5)):
    model.train(shuffle_corpus(corpus), epochs=50, total_examples=model.corpus_count)

model.save('w2v_model')

#model = Word2Vec.load('w2v_model')
#%%
#model.wv.most_similar('stark')

from sklearn.decomposition import PCA

PCA = PCA(n_components=2)
#%%
all_word_vectors_matrix = model.wv.vectors
#%%
all_word_vectors_matrix_2d = PCA.fit_transform(all_word_vectors_matrix)
#%%gen
# Create a dataframe with all the words and their coordinates
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[model.wv.vocab[word].index])
            for word in model.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
#%%
points.head(10)
