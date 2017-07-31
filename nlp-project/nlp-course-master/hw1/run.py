#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from word2vec import *
from sgd import *
from knn import *

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

startTime=time.time()
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) /
       dimVectors, np.zeros((nWords, dimVectors))),
    axis=0)
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                     negSamplingCostAndGradient),
    wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

run_time = (time.time() - startTime)
print "sanity check: cost at convergence should be around or below 10"
print "training took %d seconds" % (time.time() - startTime)

# concatenate the input and output word vectors
wordVectors = np.concatenate(
    (wordVectors[:nWords,:], wordVectors[nWords:,:]),
    axis=0)
# wordVectors = wordVectors[:nWords,:] + wordVectors[nWords:,:]

visualizeWords = [
    "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"]

visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in xrange(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('q3_word_vectors.png')

key_words = ["the", "unique", "superb", "comedy", "surprisingly"]
inputVectors = wordVectors[:nWords]
inv_tokens = {v: k for k, v in tokens.iteritems()}


import HTML

t = HTML.Table(header_row=key_words)
N = 11
rows = [[] for i in xrange(N)]

for column, key_word in enumerate(key_words):
    wordVector = inputVectors[tokens[key_word]]
    idx = knn(wordVector, inputVectors, N)
    nearest = [inv_tokens[i] for i in idx]
    print "Words related to \"" + key_word + "\": ",  nearest
    assert len(nearest) == N
    for column, neighbor in enumerate(nearest):
        rows[column].append(neighbor)

for r in rows:
    t.rows.append(r)

print str(t)
# with open("results.html", "wb") as f:
#     f.write(str(t))


l = [['UNK', 'show-don', 'movie-industry', 't-tell', 'punch-and-judy', 'tech-geeks', 'hirosue', 'sandlerian', 'loosely-connected', 'acting-workshop', 'teen-exploitation'],
['UNK', 'superficiale', 'fantasti', 'surfacey', 'prefeminist', 'brit-com', 'gone-to-seed', 'the-cash', 'phoned-in', 'director-chef', 'doing-it-for'],
['UNK', 'fleet-footed', 'revigorates', 'qutting', 'meanspirited', 'screwed-up', 'bone-dry', 'new-agey', 'kids-cute', 'handbag-clutching', 'eye-rolling'],
['UNK', 'masterpeice', 'munchausen-by-proxy', 'two-wrongs-make-a-right', 'star\\/producer', 'kid-empowerment', 'still-contemporary', 'messing-about', 'wind-in-the-hair', 'razor-sided', 'action-fantasy'],
['UNK', 'super-cool', 'smart-aleck', 're-assess', 'giant-screen', 'gabbiest', 'not-so-stock', 'feardotcom.com', 'pro-fat', 'wise-cracker', 'consciousness-raiser']]

table = [[l[i][j] for i in xrange(len(l))] for j in xrange(len(l[0]))]
t = HTML.Table(header_row=key_words)
for r in table:
    t.rows.append(r)

print str(t)
# with open("resultsGlove.html", "wb") as f:
#     f.write(str(t))
