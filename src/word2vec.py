#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

import time

import skfuzzy  as fuzz
import pandas   as pd
import numpy    as np

from gensim.models import Word2Vec


# Create bags of centroids
def create_bag_of_fuzzy_centroids(wordlist, word_centroid_map):
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = np.matrix(word_centroid_map.values()).max() + 1

    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    # Loop over the words in the text. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count by one
    for word in wordlist:
        if word in word_centroid_map:
            cluster_arr = word_centroid_map[word]
            for cl in cluster_arr:
                bag_of_centroids[cl] += 1

    # Return the "bag of centroids"
    return bag_of_centroids


def softVectorizer(train_data, test_data):
    model = Word2Vec.load(".\\models\\300features_40minwords_10context")

    # Run FCM on the word vectors and print a few clusters
    start = time.time()  # Start time

    # Set "c" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    word_vectors = word_vectors.transpose()

    num_clusters = word_vectors.shape[1] / 5
    num_clusters = 100
    print("Selected ", num_clusters, " clusters")

    # Initalize a FCM object and use it to extract centroids
    print "Running FCM"
    print "Word vector length: ", word_vectors.shape

    cntr, u, u0, d, idx, p, fpc = fuzz.cluster.cmeans(word_vectors,
                                                      num_clusters,
                                                      2,
                                                      error=0.005,
                                                      maxiter=1000,
                                                      init=None)

    # idx = np.argmax(u, axis=0)  # Hardening for visualization
    idx4five = np.argpartition(u, -4, 0)[-4:]
    idx = idx4five

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "FCM clustering during: ", elapsed, " sec"

    # Create a word2index dictionary
    # mapping each vocabulary word to a cluster number
    print len(model.index2word), idx.shape

    word_centroid_map = dict()
    i = 0
    for w in model.index2word:
        word_centroid_map[w] = idx.transpose()[i]
        i += 1

    # Print the first ten clusters
    # TODO

    # Create bags of fuzzy centroids
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros((len(train_data), num_clusters), dtype="float32")

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in train_data:
        train_centroids[counter] = create_bag_of_fuzzy_centroids(review, word_centroid_map)
        counter += 1

    # Repeat for test reviews
    test_centroids = np.zeros((len(test_data), num_clusters), dtype="float32")

    counter = 0
    for review in test_data:
        test_centroids[counter] = create_bag_of_fuzzy_centroids(review, word_centroid_map)
        counter += 1

    return train_centroids, test_centroids