#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

import csv

from src.preprocessing import convert_xml2tsv
from src.training import train
from src.testing import test


# Hardcoded input files
FILE_TRAIN_BANK = ".\\data\\SentiRuEval_twitter\\bank_train.xml"
FILE_TEST_BANK = ".\\data\\SentiRuEval_twitter\\bank_test.xml"
FILE_TRAIN_TTK = ".\\data\\SentiRuEval_twitter\\ttk_train.xml"
FILE_TEST_TTK = ".\\data\\SentiRuEval_twitter\\ttk_test.xml"
FILE_TEST_TTK_ETALON = ".\\data\\SentiRuEval_twitter\\eval\\ttk_test_etalon.xml"
FILE_TEST_BANK_ETALON = ".\\data\\SentiRuEval_twitter\\eval\\bank_test_etalon.xml"


def main(args=None):
    global mode, categories_type, train_data, categories, train_type
    file_train_bank = FILE_TRAIN_BANK
    file_test_bank = FILE_TEST_BANK
    file_train_ttk = FILE_TRAIN_TTK
    file_test_ttk = FILE_TEST_TTK
    file_test_ttk_etalon = FILE_TEST_TTK_ETALON
    file_test_bank_etalon = FILE_TEST_BANK_ETALON

    # Start
    print "Menu:"
    print "1. Convert data from xml to tsv/csv"
    print "2. Train model"
    print "3. Test model"
    print "4. Exit"

    while True:
        try:
            mode = int(raw_input("Select action: "))
            if 0 < mode < 5:
                break
        except ValueError:
            print "ERROR: select number from menu"

    if mode != 4:
        # Select categories for sentiment analysis
        print "Select categories:"
        print "1. Positive, negative"
        print "2. Positive, neutral, negative"

        while True:
            try:
                categories_type = int(raw_input("Select categories for training and predictions: "))
                if 0 < categories_type < 3:
                    break
            except ValueError:
                print "ERROR: select number from menu"

        if categories_type == 1:
            categories = ['positive', 'negative']
        elif categories_type == 2:
            categories = ['positive', 'negative', 'neutral']

    if mode == 1:
        #
        # Convert xml
        #
        convert_xml2tsv(".\\data\\raw\\")
    elif mode == 2:
        #
        # Load train data and train model
        #
        print "Run:"
        print "1. Preprocessing and training"
        print "2. Training"
        while True:
            try:
                train_type = int(raw_input("Select train type: "))
                if 0 < train_type < 3:
                    break
            except ValueError:
                print "ERROR: select number from menu"

        print "Input data:"
        print "1. Telecommunication companies"
        print "2. Banks"
        while True:
            try:
                train_data = int(raw_input("Select data type: "))
                if 0 < train_data < 3:
                    break
            except ValueError:
                print "ERROR: select number from menu"

        # TODO change
        if train_data == 1:
            train_file = ".\\data\\parsed\\ttk_train.tsv"
            test_file = ".\\data\\parsed\\ttk_test_etalon.tsv"
        elif train_data == 2:
            train_file = ".\\data\\parsed\\bank_train.tsv"
            test_file = ".\\data\\parsed\\bank_test_etalon.tsv"

        with open(train_file) as train_in, \
                open(test_file) as test_in:
            train_in = csv.reader(train_in, delimiter='\t')
            test_in = csv.reader(test_in, delimiter='\t')
            # train_in = pandas.read_csv(train_in, sep='\t', skiprows=[0], header=None)
            # test_in  = pandas.read_csv(test_in, sep='\t', skiprows=[0], header=None)

            # start = time.time()
            if train_type == 1:
                print("pre")
            elif train_type == 2:
                train(train_in, test_in, categories)
            # end = time.time()
            # print "INFO: Training during " + str(end - start) + " sec"

    elif mode == 3:
        #
        # Load model, test data and perform prediction
        #
        test(categories)

    elif mode == 4:
        #
        # Exit
        #
        print "Press any key for exit"
        exit(0)


if __name__ == '__main__':
    main()

