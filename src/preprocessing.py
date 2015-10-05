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
import xmltodict

from stemming import Porter
# from nltk.corpus import stopwords

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


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False, stemming=False):
    tokens = tokenize(s)
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
    if stemming:
        tokens = [token if emoticon_re.search(token) else Porter.stem(token) for token in tokens]
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

    file_ix = int(raw_input("Select file: "))

    if file_ix > i:
        print "Error: select file"
        exit(0)

    file_in = raw_files_path + files[file_ix]

    print "File type:"
    print "1. ttk"
    print "2. bank"
    file_type = int(raw_input("Select type: "))

    if file_type == 1:
        header = header_ttk
        header_names = header_ttk_names
    elif file_type == 2:
        header = header_bank
        header_names = header_bank_names
    else:
        print "Error: select type"
        exit(0)

    name = os.path.splitext(files[file_ix])[0]

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
            row_text = []
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

    print "Successfully converted"
