#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

import csv
import os
import re
import string
import time

import xmltodict
from pymystem3 import Mystem

from stemming import Porter


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?u)\w+',
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

m = Mystem()
prep_counter = 0


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False, stemming=False):
    # tokens = tokenize(s)
    global m
    global prep_counter
    tokens = m.lemmatize(s)

    # Print a status message every 1000th review
    if prep_counter % 100. == 0.:
        print "Lemmatize %d twits" % (prep_counter)

    prep_counter += 1
    # for t in tokens:
    #     print t
    # exit(0)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    if True:
        punctuation = list(string.punctuation)
        # stop = stopwords.words('russian') + punctuation + ['rt', 'via']
        stop = punctuation + ['rt', 'via', '...', "…".decode("utf-8")]
        # stop += stopwords.words('russian')
        # tokens1 = []
        # for token in tokens:
        #     if emoticon_re.search(token):
        #         tokens1.append(token)
        #     elif token not in stop:
        #         tokens1.append(token)
        # tokens = tokens1
        tokens = [token for token in tokens if token not in stop]

        # tokens = [token if token emoticon_re.search(token) else not in stop for token in tokens]
    # if stemming:
    #     tokens = [token if emoticon_re.search(token) else Porter.stem(token) for token in tokens]
    return tokens


#
# Converting files from Romip format 2 txt
#
def convert_xml2tsv(raw_files_path):
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

    header_ttk = ['id', 'twitid', 'date', 'text', 'beeline', 'mts',
                  'megafon', 'tele2', 'rostelecom', 'komstar', 'skylink']
    header_ttk_names = ['beeline', 'mts', 'megafon', 'tele2',
                        'rostelecom', 'komstar', 'skylink']
    header_bank = ['id', 'twitid', 'date', 'text', 'sberbank', 'vtb',
                   'gazprom', 'alfabank', 'bankmoskvy', 'raiffeisen', 'uralsib', 'rshb']
    header_bank_names = ['sberbank', 'vtb', 'gazprom', 'alfabank', 'bankmoskvy',
                         'raiffeisen', 'uralsib', 'rshb']

    i = 0
    files = os.listdir(raw_files_path)

    for f in files:
        print(str(i) + ". " + f)
        i += 1

    while True:
        try:
            file_ix = int(raw_input("Select file: "))
            if 0 <= file_ix < i:
                break
        except ValueError:
            print "ERROR: select file from list"

    file_in = raw_files_path + files[file_ix]

    print "File type:"
    print "1. Telecommunication companies"
    print "2. Banks"
    while True:
        try:
            file_type = int(raw_input("Select data type: "))
            if 0 < file_type < 3:
                break
        except ValueError:
            print "ERROR: select file from list"

    if file_type == 1:
        header = header_ttk
        header_names = header_ttk_names
    elif file_type == 2:
        header = header_bank
        header_names = header_bank_names

    name = os.path.splitext(files[file_ix])[0]
    if not os.path.exists(".\\data\\parsed\\"):
        os.makedirs(".\\data\\parsed\\")

    start = time.time()

    with open(file_in) as xml_in, \
            open(".\\data\\parsed\\" + name + "_full.tsv", "wb") as tsv_out, \
            open(".\\data\\parsed\\" + name + ".tsv", "wb") as tsv_out_text:
        obj = xmltodict.parse(xml_in.read())
        tsv_out = csv.writer(tsv_out, delimiter='\t')
        tsv_out_text = csv.writer(tsv_out_text, delimiter='\t')

        # Write header
        tsv_out.writerow(header)
        tsv_out_text.writerow(["sentiment", "text"])

        # Write data
        for table in obj['pma_xml_export']['database']["table"]:
            row = []
            for column in table["column"]:
                row.append(column["#text"].encode('utf8'))
                if column["@name"] == "text":
                    text = column["#text"].encode('utf8').replace('\n', ' ')
                if column["@name"] in header_names:
                    if column["#text"] == "0":
                        tsv_out_text.writerow(["neutral", text])
                        break  # append one time (not for each brand)
                    elif column["#text"] == "1":
                        tsv_out_text.writerow(["positive", text])
                        break
                    elif column["#text"] == "-1":
                        tsv_out_text.writerow(["negative", text])
                        break
            tsv_out.writerow(row)

    end = time.time()
    print "INFO: Converting during " + str(end - start) + " sec"
    print "Successfully converted"

if __name__ == '__main__':
    print "INFO: Stemming file: "
    i = 0
    parsed_files_path = "..\\data\\parsed\\"
    files = os.listdir(parsed_files_path)

    for f in files:
        print(str(i) + ". " + f)
        i += 1

    while True:
        try:
            file_ix = int(raw_input("Select file: "))
            if 0 <= file_ix < i:
                break
        except ValueError:
            print "ERROR: select file from list"

    file_in = parsed_files_path + files[file_ix]

    name = os.path.splitext(files[file_ix])[0]
    if not os.path.exists("..\\data\\stemmed\\"):
        os.makedirs("..\\data\\stemmed\\")

    start = time.time()

    with open(file_in) as parsed_in, \
         open("..\\data\\stemmed\\" + name + "_mystem.tsv", "wb") as mystem_out:
         # open("..\\data\\stemmed\\" + name + "_porter.tsv", "wb") as porter_out, \

        parsed_in = csv.reader(parsed_in, delimiter='\t')
        mystem_out = csv.writer(mystem_out, delimiter='\t') #, quoting=csv.QUOTE_NONE

        mystem = Mystem()
        prep_counter = 0

        for row in parsed_in:
            exclude = ['\'', '\"', '.', ',', '!', '?', u'«', u'»']
            s = ''.join(ch for ch in row[1].decode("utf-8") if ch not in exclude)

            stemmed_tokens = m.lemmatize(s)
            stemmed_tokens = [token if emoticon_re.search(token) else token.lower() for token in stemmed_tokens]

            # punctuation = list(string.punctuation.decode("utf-8"))
            # stop = punctuation
            # stop = ['!', '"', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
            #         ':', ';', '<', '=', '>', '?', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'] #'@',
            stop = ['rt', 'via', '...', "…".decode("utf-8")]
            stemmed_tokens = [token if token not in stop else '' for token in stemmed_tokens]

            stemmed_str = "".join([token for token in stemmed_tokens])
            mystem_out.writerow([row[0], stemmed_str.encode("utf-8").replace('\n', ' ')])

            # Print a status message every 1000th review
            if prep_counter % 100. == 0.:
                print "Lemmatize %d strings" % (prep_counter)

            prep_counter += 1
