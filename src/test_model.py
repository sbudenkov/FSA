#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

import csv
import json
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
import cPickle as pickle

# from train_model import preprocess
from preprocessing import preprocess

def test(categories):
    use_stem = True
    test_data = []
    test_labels = []
    tsv_out1 = open(".\\src\\test.tsv", "wb")
    tsv_out = csv.writer(tsv_out1, delimiter='\t')

    test_json = open(".\\src\\test.json")
    count_all = Counter()
    for r in test_json:
        tweet = json.loads(r)
        if (tweet["lang"] != "ru"):
            continue
        # Create a list with all the terms
        terms_all = [term for term in preprocess(tweet['text'], True)]
        # Update the counter
        count_all.update(terms_all)
        # tokens = preprocess(tweet['text'], True)
        # for token in tokens:
        #     print token

        # print tweet["text"].encode(sys.stdout.encoding, errors='replace')
        # tsv_out.writerow(["hz", tweet["text"].encode("utf-8")])
    for token in count_all.most_common(5):
        print token[0] + ":" + str(token[1])
    exit()
    tsv_out1.close()
    # exit(0)

    # test_in = open(".\\data\\parsed\\ttk_train.tsv")
    test_in = open(".\\src\\test.tsv")
    test_in  = csv.reader(test_in, delimiter='\t')

    fin1 = open('vectorizer.pk', 'r')
    vectorizer = pickle.load(fin1)

    fin2 = open('classifier_linear.pk', 'r')
    classifier_linear =  pickle.load(fin2)

    test_data, test_labels = preprocess(test_in, use_stem)
    test_vectors = vectorizer.transform(test_data)

    prediction_linear = classifier_linear.predict(test_vectors)

    print("Results for SVC(kernel=linear)")
    print(classification_report(test_labels, prediction_linear))
    with open("result_linear_test.txt", "wb") as result_out:
        i = 0
        for s in prediction_linear:
            if (test_labels[i] != prediction_linear[i]):
                result_out.write(test_labels[i] + " : " + prediction_linear[i] + '\t' + test_data[i].encode("utf-8") + '\n')
            i += 1