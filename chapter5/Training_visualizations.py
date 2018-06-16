
# coding: utf-8

# # Setup Visdom
# 
# Install it with:
# 
# `pip install visdom`
# 
# Start the server:
# 
# `python -m visdom.server`
# 
# Visdom now can be accessed at http://localhost:8097 in the browser.
# 
# 
# # LDA Training Visualization
# 
# Knowing about the progress and performance of a model, as we train them, could be very helpful in understanding it’s learning process and makes it easier to debug and optimize them. In this notebook, we will learn how to visualize training statistics for LDA topic model in gensim. To monitor the training, a list of Metrics is passed to the LDA function call for plotting their values live as the training progresses. 
# 
# 
# <img src="visdom_graph.png">
# 
# 
# Let's plot the training stats for an LDA model being trained on kaggle's [fake news dataset](https://www.kaggle.com/mrisdal/fake-news). We will use the four evaluation metrics available for topic models in gensim: Coherence, Perplexity, Topic diff and Convergence. (using separate hold_out and test corpus for evaluating the perplexity)

# In[*]

from gensim.models.fasttext import FastText
from gensim.corpora import Dictionary
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
import numpy as np

df_fake = pd.read_csv('fake.csv')
df_fake[['title', 'text', 'language']].head()
df_fake = df_fake.loc[(pd.notnull(df_fake.text)) & (df_fake.language == 'english')]

# remove stopwords and punctuations
def preprocess(row):
    return strip_punctuation(remove_stopwords(row.lower()))
    
df_fake['text'] = df_fake['text'].apply(preprocess)

# Convert data to required input format by LDA
texts = []
for line in df_fake.text:
    lowered = line.lower()
    words = re.findall(r'\w+', lowered)
    texts.append(words)


# In[*]

from gensim.test.utils import common_texts as sentences
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile

class EpochSaver(CallbackAny2Vec):
    "Callback to save model after every epoch"
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0
    def on_epoch_end(self, model):
        output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        print("Save model to {}".format(output_path))
        model.save(output_path)
        self.epoch += 1

# to save the similarity scores
similarity = []

class EpochLogger(CallbackAny2Vec):
    "Callback to log information about training"
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1
    def on_batch_begin(self, model):
        similarity.append(model.wv.similarity('woman', 'man'))


# In[*]

epoch_saver = EpochSaver(get_tmpfile("temporary_model"))
epoch_logger = EpochLogger()


# In[*]

# from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric

# # define perplexity callback for hold_out and test corpus
# # pl_holdout = PerplexityMetric(texts=texts, logger="visdom", title="Perplexity (hold_out)")
# # pl_test = PerplexityMetric(texts=texts, logger="visdom", title="Perplexity (test)")

# # define other remaining metrics available
# ch_umass = CoherenceMetric(texts=texts, coherence="u_mass", logger="visdom", title="Coherence (u_mass)")
# # ch_cv = CoherenceMetric(texts=texts, texts=training_texts, coherence="c_v", logger="visdom", title="Coherence (c_v)")
# ch_cv = CoherenceMetric(texts=texts, coherence="c_v", logger="visdom", title="Coherence (c_v)")
# diff_kl = DiffMetric(distance="kullback_leibler", logger="visdom", title="Diff (kullback_leibler)")
# convergence_kl = ConvergenceMetric(distance="jaccard", logger="visdom", title="Convergence (jaccard)")

# callbacks = [ch_umass]

# # training LDA model
# # model = ldamodel.LdaModel(corpus=training_corpus, id2word=dictionary, num_topics=35, passes=50, chunksize=1500, iterations=200, alpha='auto', callbacks=callbacks)


# In[*]

import gensim
import os
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim

# Set file names for train and test data
lee_train_file = './gensim/gensim/test/test_data/lee_background.cor'
lee_data = LineSentence(lee_train_file)

model_gensim = FT_gensim(size=100)

# build the vocabulary
model_gensim.build_vocab(lee_data)

# train the model
model_gensim.train(lee_data, 
                   total_examples=model_gensim.corpus_count,
                   epochs=model_gensim.epochs,
                   callbacks=[epoch_saver, epoch_logger])

print(model_gensim)


# When the model is set for training, you can open http://localhost:8097 to see the training progress.

# # Training Logs
# 
# We can also log the metric values after every epoch to the shell apart from visualizing them in Visdom. The only change is to define `logger="shell"` instead of `"visdom"` in the input callbacks.

# In[*]

import visdom
vis = visdom.Visdom()

trace = dict(x=[1, 2, 3], y=[4, 5, 6], mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='1st Trace')
layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})


# In[*]

len(similarity)


# In[*]

import visdom
vis = visdom.Visdom()

trace = dict(x=list(range(len(similarity))), y=similarity, mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='1st Trace')
layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})


# In[*]

similarity

