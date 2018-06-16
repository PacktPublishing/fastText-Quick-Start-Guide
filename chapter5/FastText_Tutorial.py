
# coding: utf-8

# # Using FastText via Gensim

# This tutorial is about using [fastText](https://github.com/facebookresearch/fastText) model in Gensim. There are two ways you can use fastText in Gensim - Gensim's native implementation of fastText and Gensim wrapper for fastText's original C++ code. Here, we'll learn to work with fastText library for training word-embedding models, saving & loading them and performing similarity operations & vector lookups analogous to Word2Vec.

# ## When to use FastText?
# The main principle behind fastText is that the morphological structure of a word carries important information about the meaning of the word, which is not taken into account by traditional word embeddings, which train a unique word embedding for every individual word. This is especially significant for morphologically rich languages (German, Turkish) in which a single word can have a large number of morphological forms, each of which might occur rarely, thus making it hard to train good word embeddings.  
# fastText attempts to solve this by treating each word as the aggregation of its subwords. For the sake of simplicity and language-independence, subwords are taken to be the character ngrams of the word. The vector for a word is simply taken to be the sum of all vectors of its component char-ngrams.  
# According to a detailed comparison of Word2Vec and FastText in [this notebook](Word2Vec_FastText_Comparison.ipynb), fastText does significantly better on syntactic tasks as compared to the original Word2Vec, especially when the size of the training corpus is small. Word2Vec slightly outperforms FastText on semantic tasks though. The differences grow smaller as the size of training corpus increases.
# Training time for fastText is significantly higher than the Gensim version of Word2Vec (`15min 42s` vs `6min 42s` on text8, 17 mil tokens, 5 epochs, and a vector size of 100).  
# fastText can be used to obtain vectors for out-of-vocabulary (OOV) words, by summing up vectors for its component char-ngrams, provided at least one of the char-ngrams was present in the training data.

# ## Training models

# For the following examples, we'll use the Lee Corpus (which you already have if you've installed gensim) for training our model.
# 
# For using the wrapper for fastText, you need to have fastText setup locally to be able to train models. See [installation instructions for fastText](https://github.com/facebookresearch/fastText/#requirements) if you don't have fastText installed already.

# ### Using Gensim's implementation of fastText

# In[*]

import gensim
import os
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim

# Set file names for train and test data
data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
lee_train_file = data_dir + 'lee_background.cor'
lee_data = LineSentence(lee_train_file)

model_gensim = FT_gensim(size=100)

# build the vocabulary
model_gensim.build_vocab(lee_data)

# train the model
model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)

print(model_gensim)


# ### Using wrapper for fastText's C++ code

# In[*]

from gensim.models.wrappers.fasttext import FastText as FT_wrapper

# Set FastText home to the path to the FastText executable
ft_home = '/usr/local/bin/fasttext'

# train the model
model_wrapper = FT_wrapper.train(ft_home, lee_train_file)

print(model_wrapper)


# ### Training hyperparameters

# Hyperparameters for training the model follow the same pattern as Word2Vec. FastText supports the folllowing parameters from the original word2vec - 
#      - model: Training architecture. Allowed values: `cbow`, `skipgram` (Default `cbow`)
#      - size: Size of embeddings to be learnt (Default 100)
#      - alpha: Initial learning rate (Default 0.025)
#      - window: Context window size (Default 5)
#      - min_count: Ignore words with number of occurrences below this (Default 5)
#      - loss: Training objective. Allowed values: `ns`, `hs`, `softmax` (Default `ns`)
#      - sample: Threshold for downsampling higher-frequency words (Default 0.001)
#      - negative: Number of negative words to sample, for `ns` (Default 5)
#      - iter: Number of epochs (Default 5)
#      - sorted_vocab: Sort vocab by descending frequency (Default 1)
#      - threads: Number of threads to use (Default 12)
#      
# In addition, FastText has three additional parameters - 
#     - min_n: min length of char ngrams (Default 3)
#     - max_n: max length of char ngrams (Default 6)
#     - bucket: number of buckets used for hashing ngrams (Default 2000000)
# Parameters `min_n` and `max_n` control the lengths of character ngrams that each word is broken down into while training and looking up embeddings. If `max_n` is set to 0, or to be lesser than `min_n`, no character ngrams are used, and the model effectively reduces to Word2Vec.
# 
# To bound the memory requirements of the model being trained, a hashing function is used that maps ngrams to integers in 1 to K. For hashing these character sequences, the [Fowler-Noll-Vo hashing function](http://www.isthe.com/chongo/tech/comp/fnv) (FNV-1a variant) is employed.

# **Note:** As in the case of Word2Vec, you can continue to train your model while using Gensim's native implementation of fastText. However, continuation of training with fastText models while using the wrapper is not supported.

# ## Saving/loading models

# Models can be saved and loaded via the `load` and `save` methods.

# In[*]

# saving a model trained via Gensim's fastText implementation
model_gensim.save('saved_model_gensim')
loaded_model = FT_gensim.load('saved_model_gensim')
print(loaded_model)

# saving a model trained via fastText wrapper
model_wrapper.save('saved_model_wrapper')
loaded_model = FT_wrapper.load('saved_model_wrapper')
print(loaded_model)


# The `save_word2vec_method` causes the vectors for ngrams to be lost. As a result, a model loaded in this way will behave as a regular word2vec model.  
# 

# ## Word vector lookup
# **Note:** Operations like word vector lookups and similarity queries can be performed in exactly the same manner for both the implementations of fastText so they have been demonstrated using only the fastText wrapper here.
# 
# FastText models support vector lookups for out-of-vocabulary words by summing up character ngrams belonging to the word.

# In[*]

print('night' in model_wrapper.wv.vocab)
print('nights' in model_wrapper.wv.vocab)
print(model_wrapper['night'])
print(model_wrapper['nights'])


# The word vector lookup operation only works if at least one of the component character ngrams is present in the training corpus. For example -

# In[*]

# Raises a KeyError since none of the character ngrams of the word `axe` are present in the training data
model_wrapper['axe']


# The `in` operation works slightly differently from the original word2vec. It tests whether a vector for the given word exists or not, not whether the word is present in the word vocabulary. To test whether a word is present in the training word vocabulary -

# In[*]

# Tests if word present in vocab
print("word" in model_wrapper.wv.vocab)
# Tests if vector present for word
print("word" in model_wrapper)


# ## Similarity operations

# Similarity operations work the same way as word2vec. **Out-of-vocabulary words can also be used, provided they have at least one character ngram present in the training data.**

# In[*]

print("nights" in model_wrapper.wv.vocab)
print("night" in model_wrapper.wv.vocab)
model_wrapper.wv.similarity("night", "nights")


# Syntactically similar words generally have high similarity in fastText models, since a large number of the component char-ngrams will be the same. As a result, fastText generally does better at syntactic tasks than Word2Vec. A detailed comparison is provided [here](Word2Vec_FastText_Comparison.ipynb).

# ### Other similarity operations

# In[*]

# The example training corpus is a toy corpus, results are not expected to be good, for proof-of-concept only
model_wrapper.most_similar("nights")


# In[*]

model_wrapper.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])


# In[*]

model_wrapper.doesnt_match("breakfast cereal dinner lunch".split())


# In[*]

model_wrapper.most_similar(positive=['baghdad', 'england'], negative=['london'])


# In[*]

question_file_path = data_dir + 'questions-words.txt'

model_wrapper.accuracy(questions=question_file_path)


# In[*]

# Word Movers distance
sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
sentence_president = 'The president greets the press in Chicago'.lower().split()

# Remove their stopwords.
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
sentence_obama = [w for w in sentence_obama if w not in stopwords]
sentence_president = [w for w in sentence_president if w not in stopwords]

# Compute WMD.
distance = model_wrapper.wmdistance(sentence_obama, sentence_president)
distance


# In[*]



