#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

import sys
import csv
import pandas

from preprocessing  import convert_xml2tsv
from train_model    import train
from test_model     import test

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

    #
    # Select action
    #
    while (1):
        try:
            mode = int(raw_input("Select action: "))
            if 0 < mode and mode < 5:
                break
        except ValueError:
            print "Error: select menu number"

    #
    # Select categories for sentiment analysis
    #
    print "Categories:"
    print "1. Positive, negative"
    print "2. Positive, neutral, negative"
    categories_type = int(raw_input("Select categories for training and predictions: "))

    if (categories_type == 1):
        categories = ['positive', 'negative']
    elif (categories_type == 2):
        categories = ['positive', 'negative', 'neutral']
    else:
       exit()

    #
    # Converting SentiRuEval data
    #
    if (mode == 1):
        raw_files_path = ".\\data\\raw\\"
        convert_xml2tsv(raw_files_path)

    #
    # Load train data and train model
    #
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

    #
    # Load model, test data and perform prediction
    #
    if (mode == 3):
        print "Test model ..."
        test(categories)

    #
    # Exit
    #
    if (mode == 4):
        print "Press any key for exit"
        exit(1)

    exit(0)
