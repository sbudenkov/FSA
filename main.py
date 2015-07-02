# -*- coding: utf-8 -*-
__author__ = 'Семен'

# from src import features, datalink, hashtags, plot
from src import data_proc
# TODO move to init
import time
import xml.etree.ElementTree as et
import csv
import xmltodict

# Hardcoded input files
file_train_bank = ".\\data\\SentiRuEval_twitter\\bank_train.xml"
file_test_bank = ".\\data\\SentiRuEval_twitter\\bank_test.xml"
file_train_ttk = ".\\data\\SentiRuEval_twitter\\ttk_train.xml"
file_test_ttk = ".\\data\\SentiRuEval_twitter\\ttk_test.xml"

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
    file_in = file_train_ttk

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
         open(".\\data\\parsed\\ttk_train.tsv", "wb") as tsv_out_text:
        obj = xmltodict.parse(xml_in.read())
        tsv_out = csv.writer(tsv_out, delimiter='\t')
        tsv_out_text = csv.writer(tsv_out_text, delimiter='\t')

        # Write header
        tsv_out.writerow(header_ttk)
        tsv_out_text.writerow(["sentiment", "text"])

        # Write data
        for table in obj['pma_xml_export']['database']["table"]:
            row = []
            row_text = []
            for column in table["column"]:
                row.append(column["#text"].encode('utf8'))
                if column["@name"] == "text":
                    # row_text.append(column["#text"].encode('utf8'))
                    text = column["#text"].encode('utf8')
                if column["@name"] in header_ttk_names:
                    if column["#text"] == "0":
                        tsv_out_text.writerow(["neutral", text])
                    elif column["#text"] == "1":
                        # row_text.insert(0, "positive")
                        tsv_out_text.writerow(["positive", text])
                    elif column["#text"] == "-1":
                        # row_text.insert(0, "negative")
                        tsv_out_text.writerow(["negative", text])
                # print column["@name"] + '\t' + column["#text"]
            tsv_out.writerow(row)



    # print obj['pma_xml_export']['database']['@name']
    # writer.close()

# Load train data and train model
if (mode == 2):
    print "Train model"

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
