# -*- coding: utf-8 -*-
__author__ = 'Семен'

# from src import features, datalink, hashtags, plot
from src import data_proc
# TODO move to init
import sys
import os
import time
import csv
import xmltodict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report

def usage():
    print("Usage:")
    print("python %s <data_dir>" % sys.argv[0])

if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     usage()
    #     sys.exit(1)

    # Hardcoded input files
    file_train_bank = ".\\data\\SentiRuEval_twitter\\bank_train.xml"
    file_test_bank = ".\\data\\SentiRuEval_twitter\\bank_test.xml"
    file_train_ttk = ".\\data\\SentiRuEval_twitter\\ttk_train.xml"
    file_test_ttk = ".\\data\\SentiRuEval_twitter\\ttk_test.xml"
    file_test_ttk_etalon = ".\\data\\SentiRuEval_twitter\\eval\\ttk_test_etalon.xml"
    file_test_bank_etalon = ".\\data\\SentiRuEval_twitter\\eval\\bank_test_etalon.xml"

    # Entry point
    print "Menu:"
    print "1. Convert data from xml to tsv/csv"
    print "2. Train model"
    print "3. Test model"

    # Default mode
    mode = 0

    while (1):
        try:
            mode = int(raw_input("Select action: "))
            if 0 < mode and mode < 4:
                break
        except ValueError:
            print "Not a menu item"

    # Converting files from Romip format 2 txt
    """
        Bank
        `id` int(11) NOT NULL AUTO_INCREMENT,
        `twitid` bigint(32) NOT NULL DEFAULT '0',
        `date` varchar(128) DEFAULT NULL,
        `text` varchar(256) DEFAULT NULL,
        `sberbank` int(11) DEFAULT NULL,
        `vtb` int(11) DEFAULT NULL,
        `gazprom` int(11) DEFAULT NULL,
        `alfabank` int(11) DEFAULT NULL,
        `bankmoskvy` int(11) DEFAULT NULL,
        `raiffeisen` int(11) DEFAULT NULL,
        `uralsib` int(11) DEFAULT NULL,
        `rshb` int(11) DEFAULT NULL,
        PRIMARY KEY (`id`),
        UNIQUE KEY `id` (`twitid`)
    """
    """
        TTK
        `id` int(11) NOT NULL AUTO_INCREMENT,
        `twitid` bigint(32) NOT NULL DEFAULT '0',
        `date` varchar(128) DEFAULT NULL,
        `text` varchar(256) DEFAULT NULL,
        `beeline` varchar(10) DEFAULT NULL,
        `mts` varchar(10) DEFAULT NULL,
        `megafon` varchar(10) DEFAULT NULL,
        `tele2` varchar(10) DEFAULT NULL,
        `rostelecom` varchar(10) DEFAULT NULL,
        `komstar` varchar(10) DEFAULT NULL,
        `skylink` varchar(10) DEFAULT NULL,
        PRIMARY KEY (`id`),
        UNIQUE KEY `id` (`twitid`),
        UNIQUE KEY `id_2` (`twitid`)
    """

    if (mode == 1):
        # file_in = str(raw_input("Please select input file: \n"))
        file_in = file_test_bank_etalon

        # Define variables
        header_ttk = ['id', 'twitid', 'date', 'text', 'beeline',
                       'mts', 'megafon', 'tele2', 'rostelecom',
                       'komstar', 'skylink']
        header_ttk_names = ['beeline', 'mts', 'megafon', 'tele2', 'rostelecom',
                            'komstar', 'skylink']

        header_bank = ['id', 'twitid', 'date', 'text', 'sberbank',
                       'vtb', 'gazprom', 'alfabank', 'bankmoskvy',
                       'raiffeisen', 'uralsib', 'rshb']
        header_bank_names = ['sberbank', 'vtb', 'gazprom', 'alfabank', 'bankmoskvy',
                             'raiffeisen', 'uralsib', 'rshb']

        with open(file_in) as xml_in, \
             open("out.tsv", "wb") as tsv_out, \
             open(".\\data\\parsed\\bank_test_etalon.tsv", "wb") as tsv_out_text:
            obj = xmltodict.parse(xml_in.read())
            tsv_out = csv.writer(tsv_out, delimiter='\t')
            tsv_out_text = csv.writer(tsv_out_text, delimiter='\t')

            # Write header
            tsv_out.writerow(header_bank)
            tsv_out_text.writerow(["sentiment", "text"])

            # Write data
            for table in obj['pma_xml_export']['database']["table"]:
                row = []
                row_text = []
                for column in table["column"]:
                    row.append(column["#text"].encode('utf8'))
                    if column["@name"] == "text":
                        text = column["#text"].encode('utf8')
                    if column["@name"] in header_bank_names:
                        if column["#text"] == "0":
                            tsv_out_text.writerow(["neutral", text])
                        elif column["#text"] == "1":
                            tsv_out_text.writerow(["positive", text])
                        elif column["#text"] == "-1":
                            tsv_out_text.writerow(["negative", text])
                tsv_out.writerow(row)

        print "Successfully converted"

    # Load train data and train model
    if (mode == 2):
        print "Train model"
        # data_dir = sys.argv[1]
        classes = ['pos', 'neg', 'net']

        # Read the data
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        # for curr_class in classes:
        #     dirname = os.path.join(data_dir, curr_class)
        #     for fname in os.listdir(dirname):
        #         with open(os.path.join(dirname, fname), 'r') as f:
        #             content = f.read()
        #             if fname.startswith('cv9'):
        #                 test_data.append(content)
        #                 test_labels.append(curr_class)
        #             else:
        #                 train_data.append(content)
        t0 = time.time()
        with open(".\\data\\parsed\\bank_train.tsv") as train_in, \
            open(".\\data\\parsed\\bank_test_etalon.tsv") as test_in:
            train_in = csv.reader(train_in, delimiter='\t')
            test_in  = csv.reader(test_in, delimiter='\t')
            for row in train_in:
                if row[0] == "positive":
                    train_data.append(row[1])
                    train_labels.append("pos")
                elif row[0] == "negative":
                    train_data.append(row[1])
                    train_labels.append("neg")
                elif row[0] == "neutral":
                    train_data.append(row[1])
                    train_labels.append("net")
            for row in test_in:
                if row[0] == "positive":
                    test_data.append(row[1])
                    test_labels.append("pos")
                elif row[0] == "negative":
                    test_data.append(row[1])
                    test_labels.append("neg")
                elif row[0] == "neutral":
                    test_data.append(row[1])
                    test_labels.append("net")
        t1 = time.time()
        time_load_data = t1-t0
        print "Load data in: " + str(time_load_data) + "s"

        # Create feature vectors
        vectorizer = TfidfVectorizer(min_df=5,
                                     max_df = 0.8,
                                     sublinear_tf=True,
                                     use_idf=True)
        train_vectors = vectorizer.fit_transform(train_data)
        test_vectors = vectorizer.transform(test_data)

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
        print("Results for SVC(kernel=linear)")
        print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
        print(classification_report(test_labels, prediction_linear))
        print("Results for LinearSVC()")
        print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
        print(classification_report(test_labels, prediction_liblinear))

    # Load model, test data and perform prediction
    if (mode == 3):
        print "Test model"

    exit()

    # dblink = datalink.DatabaseConnectionDown('perilipsi_tweets')
    # plotter = plot.Plotter()
    # emoTest = features.Emoticons()
    # dictTest = features.DictionaryTest()
    # hashtest = hashtags.hashtags()
    # # You can pass anything you want
    # testTweet, tweetTime = dblink.fetchTweet(
    # )['tweet'], dblink.fetchTweet()['time']
    # emo_test = emoTest.analyse(testTweet)
    # dict_test = dictTest.analyse(testTweet)
    # hash_test = hashtest.analyseHashtagTweet(testTweet)
    # print "Emoticons:", emo_test
    # print "DictionaryTest:", dict_test
    # print "Hashtags: ", hash_test
