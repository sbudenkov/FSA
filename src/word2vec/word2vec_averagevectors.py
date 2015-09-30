#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Parts 2 and 3 of the tutorial, which cover how to
#  train a model using Word2Vec.
#
# *************************************** #


# ****** Read the two training sets and the test set
#
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from KaggleWord2VecUtility import KaggleWord2VecUtility, FuzzyWord2VecUtility


# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    if (nwords != 0):
        featureVec = np.divide(featureVec, nwords)
    else:
        featureVec = np.zeros((num_features,), dtype="float32")

    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        #
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))
        #
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        #
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["text"]:
        # clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
        clean_reviews.append(FuzzyWord2VecUtility.text_to_wordlist(review, remove_stopwords=True))
    return clean_reviews


if __name__ == '__main__':
    # Read data from files
    # train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    # test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    # unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
    train = pd.read_csv('..\\..\\data\\parsed\\ttk_train.tsv',
                        header=0,
                        delimiter="\t",
                        quoting=3)
    test = pd.read_csv('..\\..\\data\\parsed\\ttk_test_etalon.tsv',
                       header=0,
                       delimiter="\t",
                       quoting=3)
    unlabeled_train = pd.read_csv('..\\..\\data\\parsed\\ttk_test.tsv',
                                  header=0,
                                  delimiter="\t",
                                  quoting=3)
    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
          "and %d unlabeled reviews\n" % (train["text"].size,
                                          test["text"].size, unlabeled_train["text"].size)



    # Load the punkt tokenizer
    # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    for review in train["text"]:
        # sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        sentences.append(FuzzyWord2VecUtility.text_to_wordlist(review, True))

    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train["text"]:
        # sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        sentences.append(FuzzyWord2VecUtility.text_to_wordlist(review, True))

    # print(len(sentences))
    # print(sentences[0])
    # exit(0)
    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 300      # Word vector dimensionality
    min_word_count = 40     # Minimum word count 40
    num_workers = 4         # Number of threads to run in parallel
    context = 10            # Context window size 10
    downsampling = 1e-3     # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers,
                     size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "..\\..\\models\\300features_40minwords_10context"
    model.save(model_name)

    # model.doesnt_match(u"мтс билайн рыба".split())
    # model.doesnt_match("france england germany berlin".split())
    # model.doesnt_match("paris berlin london austria".split())
    # model.most_similar(u"мтс")
    # model.most_similar("queen")
    # model.most_similar("awful")
    # for a in model.most_similar(u"мтс"):
    #     print(a)
    # decoded = [[word for word in sets if isinstance(word, basestring)] for sets in model.most_similar(u"мтс")]
    # for words in decoded:
    #     print " ".join(words)
    #     # print words
    # exit(0)

    # ****** Create average vectors for the training and test sets
    #
    print "Creating average feature vecs for training reviews"

    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)

    print "Creating average feature vecs for test reviews"

    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)


    # ****** Fit a random forest to the training set, then make predictions
    #
    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    print "Fitting a random forest to labeled training data..."
    # print(trainDataVecs.shape)
    # np.savetxt('train.out', trainDataVecs, delimiter='\t')
    # np.savetxt('test.out', testDataVecs, delimiter='\t')
    # with open("trainDataVecs.txt", "wb") as result_out:
    # # i = 0
    # # for s in prediction_linear:
    # #     if (test_labels[i] != prediction_linear[i]):
    # #         result_out.write(test_labels[i] + " : " + prediction_linear[i] + '\t' + test_data[i].encode("utf-8") + '\n')
    # #     i += 1
    #     result_out.write(trainDataVecs)
    # exit(0)
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # Test & extract results
    result = forest.predict(testDataVecs)

    # Write the test results
    output = pd.DataFrame(data={"target": test["sentiment"], "sentiment": result})
    print(classification_report(test["sentiment"], result))

    output.to_csv("..\\..\\results\\Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print "Wrote Word2Vec_AverageVectors.csv"
