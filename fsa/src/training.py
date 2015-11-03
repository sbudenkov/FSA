#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

import time
import string
import cPickle as pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# from stemming       import Porter
from preprocessing import preprocess


# from word2vec import softVectorizer

# from factory import create_factory, train_factory, test_factory


def stem(str_in):
    str = "".join([ch for ch in str_in if ch not in string.punctuation + "..."])
    str = str.split()
    res = []
    for word in str:
        # new_word = Porter.stem(word)
        new_word = word
        res.append(new_word)
    return " ".join([ch for ch in res])


def prepare_data(f, categories, lowercase=True, stemming=False):
    data = []
    labels = []
    for row in f:
        if row[0] in categories:
            # if (ctg == "negative"):
            #     tokens = preprocess(row[1].decode("utf-8"), lowercase, False)
            # else:
            tokens = preprocess(row[1].decode("utf-8"), lowercase, stemming)
            new_str = " ".join([token for token in tokens])
            data.append(new_str)
            if (row[0] == 'negative'):
                labels.append(-1)
            elif (row[0] == 'neutral'):
                labels.append(0)
            elif (row[0] == 'positive'):
                labels.append(1)

    return data, labels


# Train model with pipeline and grid
def train(train_in, test_in, categories):
    print "Train model for categories: ", categories
    use_stem = False  # move to option

    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    t0 = time.time()

    train_data, train_labels = prepare_data(train_in, categories, stemming=use_stem)
    test_data, test_labels = prepare_data(test_in, categories, stemming=use_stem)

    duration = time.time() - t0
    print "Tokenizing process during: " + str(duration) + "sec"

    # Save text data 2 file 4 viewing
    with open(".\\results\\train_data.txt", "wb") as result_out:
        i = 0
        for s in train_data:
            result_out.write(str(train_labels[i]) + '\t' + train_data[i].encode("utf-8") + '\n')
            i += 1

    with open(".\\results\\test_data.txt", "wb") as result_out:
        i = 0
        for s in test_data:
            result_out.write(str(test_labels[i]) + '\t' + test_data[i].encode("utf-8") + '\n')
            i += 1

    #
    # Select categories for sentiment analysis
    #
    print "Models:"
    print "1. BoW"
    print "2. TF/IDF"
    print "3. BoW TF/IDF"
    print "4. word2vec"
    print "5. word2vec k-means"
    print "6. word2vec FCM"
    model_type = int(raw_input("Select model for training and predictions: "))

    if (model_type == 2):
        print('TF/IDF')

        # Create feature vectors
        vectorizer = TfidfVectorizer(min_df=2,
                                     max_df=0.8,
                                     sublinear_tf=True,
                                     use_idf=True)  # ,
        # ngram_range=(0,2), analyzer='word')

        train_vectors = vectorizer.fit_transform(train_data)
        test_vectors = vectorizer.transform(test_data)

        print 'Train data shape after TF-IDF: ', train_vectors.shape
        print 'Test data shape after TF-IDF: ', test_vectors.shape

        # Save vectorizer for future
        with open('.\\models\\vectorizer.pk', 'wb') as fin:
            pickle.dump(vectorizer, fin)

    elif (model_type == 6):
        print('word2vec FCM')
        # train_vectors, test_vectors = softVectorizer(train_data, test_data)
    else:
        exit()


    # print train_vectors.shape, len(train_data)
    # exit()
    # train_labels = [0 for k in train_labels if k == 'negative']

    # FACTORY try later
    # factory = create_factory(None)
    # factory = train_factory(factory, DataFrame(train_vectors.todense()), train_labels, None)
    # test_factory(factory, DataFrame(test_vectors.todense()), test_labels)
    # # factory = train_factory(factory, train_vectors, train_labels, None)
    # # test_factory(factory, test_vectors, test_labels)
    #
    # exit()

    # Perform classification with naïve Bayes classifier
    classifier_nb = MultinomialNB()
    t0 = time.time()
    classifier_nb.fit(train_vectors, train_labels)
    with open('.\\models\\classifier_nb.pk', 'wb') as fin:
        pickle.dump(classifier_nb, fin)
    t1 = time.time()
    prediction_nb = classifier_nb.predict(test_vectors)
    t2 = time.time()
    time_nb_train = t1 - t0
    time_nb_predict = t2 - t1

    # Perform classification with SGDClassifier
    classifier_sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    t0 = time.time()
    classifier_sgd.fit(train_vectors, train_labels)
    with open('.\\models\\classifier_sgd.pk', 'wb') as fin:
        pickle.dump(classifier_sgd, fin)
    t1 = time.time()
    prediction_sgd = classifier_sgd.predict(test_vectors)
    t2 = time.time()
    time_sgd_train = t1 - t0
    time_sgd_predict = t2 - t1

    # Perform classification with SVM, kernel=rbf
    classifier_rbf = SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    with open('.\\models\\classifier_rbf.pk', 'wb') as fin:
        pickle.dump(classifier_rbf, fin)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1 - t0
    time_rbf_predict = t2 - t1

    # Perform classification with SVM, kernel=linear
    classifier_linear = SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    with open('.\\models\\classifier_linear.pk', 'wb') as fin:
        pickle.dump(classifier_linear, fin)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1 - t0
    time_linear_predict = t2 - t1

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    with open('.\\models\\classifier_liblinear.pk', 'wb') as fin:
        pickle.dump(classifier_liblinear, fin)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1 - t0
    time_liblinear_predict = t2 - t1

    # Perform classification with Random Forest
    classifier_forest = RandomForestClassifier(n_estimators=100)
    t0 = time.time()
    # classifier_forest.fit(train_vectors.todense(), train_labels)
    classifier_forest.fit(train_vectors, train_labels)
    with open('.\\models\\classifier_forest.pk', 'wb') as fin:
        pickle.dump(classifier_forest, fin)
    t1 = time.time()
    # prediction_forest = classifier_forest.predict(test_vectors.todense())
    prediction_forest = classifier_forest.predict(test_vectors)
    t2 = time.time()
    time_forest_train = t1 - t0
    time_forest_predict = t2 - t1

    # Print results in a nice table
    print("Results for NB")
    print("Training time: %fs; Prediction time: %fs" % (time_nb_train, time_nb_predict))
    print(classification_report(test_labels, prediction_nb))
    i = 0
    count = 0
    for s in prediction_nb:
        if (test_labels[i] != prediction_nb[i]):
            print("real: " + str(test_labels[i]) + "\t predicted: " +
                  str(prediction_nb[i]) + '\t' +
                  test_data[i] + '\n')
            count += 1

            if (count > 6):
                break

        i += 1

    print("Results for SGD")
    print("Training time: %fs; Prediction time: %fs" % (time_sgd_train, time_sgd_predict))
    print(classification_report(test_labels, prediction_sgd))
    i = 0
    count = 0
    for s in prediction_sgd:
        if (test_labels[i] != prediction_sgd[i]):
            print("real: " + str(test_labels[i]) + "\t predicted: " +
                  str(prediction_sgd[i]) + '\t' +
                  test_data[i] + '\n')
            count += 1

            if (count > 6):
                break

        i += 1

    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    # with open("result_rbf.txt", "wb") as result_out:
    i = 0
    count = 0
    for s in prediction_rbf:
        if (test_labels[i] != prediction_rbf[i]):
            print("real: " + str(test_labels[i]) + "\t predicted: " +
                  str(prediction_rbf[i]) + '\t' +
                  test_data[i] + '\n')
            count += 1

            if (count > 6):
                break

        i += 1

    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    i = 0
    count = 0
    for s in prediction_linear:
        if (test_labels[i] != prediction_linear[i]):
            print("real: " + str(test_labels[i]) + "\t predicted: " +
                  str(prediction_linear[i]) + '\t' +
                  test_data[i] + '\n')
            count += 1

            if (count > 6):
                break

        i += 1

    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))
    i = 0
    count = 0
    for s in prediction_liblinear:
        if (test_labels[i] != prediction_liblinear[i]):
            print("real: " + str(test_labels[i]) + "\t predicted: " +
                  str(prediction_liblinear[i]) + '\t' +
                  test_data[i] + '\n')
            count += 1

            if (count > 6):
                break

        i += 1

    print("Results for RandomForestClassifier()")
    print("Training time: %fs; Prediction time: %fs" % (time_forest_train, time_forest_predict))
    print(classification_report(test_labels, prediction_forest))
    i = 0
    count = 0
    for s in prediction_forest:
        if (test_labels[i] != prediction_forest[i]):
            print("real :" + str(test_labels[i]) + "\t predicted: " +
                  str(prediction_forest[i]) + '\t' +
                  test_data[i] + '\n')
            count += 1

            if (count > 6):
                break

        i += 1
        # result_file = str(len(categories)) + 'classes_results.txt'
        # with open(".\\results\\" + result_file, "wb") as result_out:
        #     result_out.write(classification_report(test_labels, prediction_nb))
        #
