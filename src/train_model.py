# -*- coding: utf-8 -*-
import time
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
import cPickle as pickle

from stemming import Porter

def stem(str_in):
    str = "".join([ch for ch in str_in if ch not in string.punctuation + "..."])
    str = str.split()
    res = []
    for word in str:
        new_word = Porter.stem(word)
        res.append(new_word)
    return " ".join([ch for ch in res])

def preprocess(file, stemming=False):
    data = []
    labels = []
    for row in file:
        if row[0] == "positive":
            if stemming:
                str1 = stem(row[1].decode("utf-8").lower())
            else:
                str1 = row[1].decode("utf-8").lower()
            data.append(str1)
            labels.append("pos")
        elif row[0] == "negative":
            # str1 = stem(row[1].decode("utf-8").lower())
            str1 = row[1].decode("utf-8").lower()
            data.append(str1)
            labels.append("neg")
        elif row[0] == "hz":
            # str1 = stem(row[1].decode("utf-8").lower())
            if stemming:
                str1 = stem(row[1].decode("utf-8").lower())
            else:
                str1 = row[1].lower()
            data.append(str1)
            labels.append("hz")
        # elif row[0] == "neutral":
        #     train_data.append(row[1])
        #     train_labels.append("net")
    return data, labels

def train(train_in, test_in):
    print "Train model ..."
    classes = ['pos', 'neg']#, 'net']
    use_stem = True
    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    t0 = time.time()

    train_data, train_labels = preprocess(train_in, use_stem)
    test_data, test_labels = preprocess(test_in, use_stem)

    t1 = time.time()
    time_load_data = t1-t0
    print "Pre process: " + str(time_load_data) + "s"

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=2,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)#,
                                 #ngram_range=(0,2), analyzer='word')

    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)
    with open('vectorizer.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)
    # Perform classification with SVM, kernel=rbf
    classifier_rbf = SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_linear = SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    with open('classifier_linear.pk', 'wb') as fin:
        pickle.dump(classifier_linear, fin)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1

    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    with open("result_rbf.txt", "wb") as result_out:
        i = 0
        for s in prediction_rbf:
            if (test_labels[i] != prediction_rbf[i]):
                result_out.write(test_labels[i] + " : " + prediction_rbf[i] + '\t' + test_data[i].encode("utf-8") + '\n')
            i += 1
    # exit(0)
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    with open("result_linear.txt", "wb") as result_out:
        i = 0
        for s in prediction_linear:
            if (test_labels[i] != prediction_linear[i]):
                result_out.write(test_labels[i] + " : " + prediction_linear[i] + '\t' + test_data[i].encode("utf-8") + '\n')
            i += 1
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))
    with open("result_liblinear.txt", "wb") as result_out:
        i = 0
        for s in prediction_liblinear:
            if (test_labels[i] != prediction_liblinear[i]):
                result_out.write(test_labels[i] + " : " + prediction_liblinear[i] + '\t' + test_data[i].encode("utf-8") + '\n')
            i += 1