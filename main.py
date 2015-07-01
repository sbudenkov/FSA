# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET

__author__ = 'Семен'

# from src import features, datalink, hashtags, plot
from src import data_proc
import time
import xml.etree.ElementTree as et
import csv
import xmltodict

file_train = "C:\\proj\\fsa\\data\\SentiRuEval_twitter\\bank_train.xml"
file_test = "./data/SentiRuEval_twitter/bank_test.xml"

print
"Menu:"
print
"1. Convert data from xml to tsv/csv"
print
"2. Train model"
print
"3. Test model"

mode = 0

while (1):
    try:
        mode = int(raw_input('Select action: \n'))
        if mode > 0 and mode < 4:
            break
    except ValueError:
        print
        "Not a menu item"

if (mode == 1):
    # file_in = str(raw_input("Please select input file: \n"))
    file_in = file_train
    xmltext = """
                <dicts>
                    <key>1375</key>
                    <dict>
                        <key>Key 1</key><integer>1375</integer>
                        <key>Key 2</key><string>Some String</string>
                        <key>Key 3</key><string>Another string</string>
                        <key>Key 4</key><string>Yet another string</string>
                        <key>Key 5</key><string>Strings anyone?</string>
                    </dict>
                </dicts>
                """

    f = open('output.txt', 'w')

    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    xmltext = open(file_in, "r").read()
    tree = et.fromstring(xmltext)



    # iterate over the dict elements
    # for dict_el in tree.iterfind('table'):
    #     data = []
    #     # get the text contents of each non-key element
    #     for el in dict_el:
    #         data.append(el.text)
    #         # if el.tag == 'string':
    #         #     data.append(el.text)
    #         # # if it's an integer element convert to int so csv wont quote it
    #         # elif el.tag == 'integer':
    #         #     data.append(int(el.text))
    #     writer.writerow(data)

    with open(file_in) as fd:
        obj = xmltodict.parse(fd.read())

    print
    obj['pma_xml_export']['database']['table']['@name']
    # writer.close()

if (mode == 2):
    print
    "Train model"

if (mode == 3):
    print
    "Test model"

exit()

tree = ET.ElementTree()  # instantiate an object of *class* `ElementTree`
tree.parse(file_train)
root = tree.getroot()
print
root
exit()

data_proc.PrepareData(file_train)
print
"Exit"
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
