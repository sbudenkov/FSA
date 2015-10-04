#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

# Load a pre-trained model
import  os
import  time
import  re

import  skfuzzy as fuzz
import  pandas as pd
import  numpy as np

from    gensim.models import Word2Vec
from    nltk.corpus import stopwords
from    sklearn.cluster import KMeans
from    sklearn.metrics import classification_report
from    sklearn.ensemble import RandomForestClassifier
from    bs4 import BeautifulSoup

from    KaggleWord2VecUtility import KaggleWord2VecUtility, FuzzyWord2VecUtility

# Create bags of centroids
def create_bag_of_centroids(wordlist, word_centroid_map):
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    # num_centroids = max(word_centroid_map.values()) + 1
    num_centroids = np.matrix(word_centroid_map.values()).max() + 1

    # print num_centroids
    # exit()

    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            # print index
            # exit()
            for ix in index:
                bag_of_centroids[ix] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


if __name__ == '__main__':
    model = Word2Vec.load("..\\..\\models\\300features_40minwords_10context")

    # Run FCM on the word vectors and print a few clusters
    start = time.time()  # Start time

    # Set "c" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    word_vectors = word_vectors.transpose()
    # for i in xrange(0, 10):
    #     print word_vectors[i]
    # exit(0)
    num_clusters = word_vectors.shape[1] / 5
    num_clusters = 100
    print num_clusters

    # Initalize a FCM object and use it to extract centroids
    print "Running FCM"
    print "Word vector length: ", word_vectors.shape
    # kmeans_clustering = KMeans( n_clusters = num_clusters )
    # idx = kmeans_clustering.fit_predict( word_vectors )
    cntr, u, u0, d, idx, p, fpc = fuzz.cluster.cmeans(word_vectors,
                                                      num_clusters,
                                                      2,
                                                      error=0.005,
                                                      maxiter=1000,
                                                      init=None)
    idx = np.argmax(u, axis=0)  # Hardening for visualization
    idx4five = np.argpartition(u, -4, 0)[-4:]
    idx = idx4five
    # print len(idx), idx
    # print u.shape, np.argwhere(u == np.amax(u)), idx4five.shape, idx4five[:, :2]
    # exit(0)

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for C Means clustering: ", elapsed, "seconds."


    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    print len(model.index2word), idx.shape

    # word_centroid_map = dict(zip(model.index2word, idx))
    word_centroid_map = dict()
    i = 0
    for w in model.index2word:
        # print w, idx.transpose()[i]
        # exit()
        word_centroid_map[w] = idx.transpose()[i]
        i += 1
    # Print the first ten clusters
    # for cluster in xrange(0, 10):
    #     # Print the cluster number
    #     print "\nCluster %d" % cluster
    #
    #     # Find all of the words for that cluster number, and print them out
    #     words = []
    #     for i in xrange(0, len(word_centroid_map.values())):
    #         # print word_centroid_map.values()[i]
    #         # exit(0)
    #         # print word_centroid_map.values()[i]
    #         if (word_centroid_map.values()[i] == cluster):
    #             # v = word_centroid_map.values()[i] == cluster
    #             # print v
    #             # # if v.any():
    #             # exit(0)
    #             # if(v.any()):
    #             #     print v
    #             words.append(word_centroid_map.keys()[i])
    #     print words


    # Create clean_train_reviews and clean_test_reviews as we did before
    # Read data from files
    # train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    # test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    train = pd.read_csv('..\\..\\data\\parsed\\ttk_train.tsv',
                        header=0,
                        delimiter="\t",
                        quoting=3)
    test = pd.read_csv('..\\..\\data\\parsed\\ttk_test_etalon.tsv',
                       header=0,
                       delimiter="\t",
                       quoting=3)

    print "Cleaning training reviews"
    clean_train_reviews = []
    for review in train["text"]:
        # clean_train_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
        clean_train_reviews.append(FuzzyWord2VecUtility.text_to_wordlist(review, remove_stopwords=True))

    print "Cleaning test reviews"
    clean_test_reviews = []
    for review in test["text"]:
        # clean_test_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
        clean_test_reviews.append(FuzzyWord2VecUtility.text_to_wordlist(review, remove_stopwords=True))


    # Create bags of centroids
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros((train["text"].size, num_clusters), dtype="float32")

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # Repeat for test reviews
    test_centroids = np.zeros((test["text"].size, num_clusters), dtype="float32")

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1


    # ****** Fit a random forest and extract predictions
    forest = RandomForestClassifier(n_estimators=100)

    # Fitting the forest may take a few minutes
    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)

    # Write the test results
    output = pd.DataFrame(data={"target": test["sentiment"], "sentiment": result})
    print(classification_report(test["sentiment"], result))

    output.to_csv("..\\..\\results\\BagOfFuzzyCentroids.csv", index=False, quoting=3)
    print "Wrote BagOfFuzzyCentroids.csv"
