# -*- coding: utf-8 -*-

import sys
import os
import csv
import xmltodict
import pandas

from train_model import train
from test_model import test

def usage():
    print("Usage:")
    print("python %s <data_dir>" % sys.argv[0])

if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     usage()
    #     sys.exit(1)

    # Hardcoded input files
    # file_train_bank         = ".\\data\\SentiRuEval_twitter\\bank_train.xml"
    # file_test_bank          = ".\\data\\SentiRuEval_twitter\\bank_test.xml"
    # file_train_ttk          = ".\\data\\SentiRuEval_twitter\\ttk_train.xml"
    # file_test_ttk           = ".\\data\\SentiRuEval_twitter\\ttk_test.xml"
    # file_test_ttk_etalon    = ".\\data\\SentiRuEval_twitter\\eval\\ttk_test_etalon.xml"
    # file_test_bank_etalon   = ".\\data\\SentiRuEval_twitter\\eval\\bank_test_etalon.xml"
    file_train_bank         = ".\\data\\SentiRuEval_twitter\\bank_train.xml"
    file_test_bank          = ".\\data\\SentiRuEval_twitter\\bank_test.xml"
    file_train_ttk          = ".\\data\\SentiRuEval_twitter\\ttk_train.xml"
    file_test_ttk           = ".\\data\\SentiRuEval_twitter\\ttk_test.xml"
    file_test_ttk_etalon    = ".\\data\\SentiRuEval_twitter\\eval\\ttk_test_etalon.xml"
    file_test_bank_etalon   = ".\\data\\SentiRuEval_twitter\\eval\\bank_test_etalon.xml"

    # Entry point
    print "Menu:"
    print "1. Convert data from xml to tsv/csv"
    print "2. Train model"
    print "3. Test model"
    print "4. Exit"

    # Default mode
    mode = 0

    while (1):
        try:
            mode = int(raw_input("Select action: "))
            if 0 < mode and mode < 5:
                break
        except ValueError:
            print "Error: select menu number"

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

    # Converting SentiRuEval data
    if (mode == 1):
        raw_files_path = ".\\data\\raw\\"
        header_ttk = ['id', 'twitid', 'date', 'text', 'beeline','mts',
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
                            break #append one time (not for each brand)
                        elif column["#text"] == "1":
                            tsv_out_text.writerow(["positive", text])
                            break
                        elif column["#text"] == "-1":
                            tsv_out_text.writerow(["negative", text])
                            break
                tsv_out.writerow(row)

        print "Successfully converted"

    # Categories for training and predictions
    # categories = ['positive', 'negative']
    categories = ['positive', 'negative', 'neutral']

    # Load train data and train model
    if (mode == 2):
        print "Run:"
        print "1. pre + train"
        print "2. train"
        train_type = int(raw_input("Select process: "))
        print "Input data:"
        print "1. ttk"
        print "2. bank"
        train_data = int(raw_input("Select data: "))

        if (train_data == 1):
            train_file = ".\\data\\parsed\\ttk_train.tsv"
            test_file  = ".\\data\\parsed\\ttk_test_etalon.tsv"
        elif (train_data == 2):
            train_file = ".\\data\\parsed\\bank_train.tsv"
            test_file  = ".\\data\\parsed\\bank_test_etalon.tsv"
        else:
            exit()

        with open(train_file) as train_in, \
            open(test_file) as test_in:
            train_in = csv.reader(train_in, delimiter='\t')
            test_in  = csv.reader(test_in, delimiter='\t')
            # train_in = pandas.read_csv(train_in, sep='\t', skiprows=[0], header=None)
            # test_in  = pandas.read_csv(test_in, sep='\t', skiprows=[0], header=None)

            if (train_type == 1):
                print("pre")
            elif (train_type == 2):
                train(train_in, test_in, categories)
            else:
                exit()

    # Load model, test data and perform prediction
    if (mode == 3):
        print "Test model ..."
        test(categories)

    if (mode == 4):
        print "Press any key for exit"
        exit(1)

    exit(0)
