# -*- coding: utf-8 -*-
__author__ = 'Семен'
from xml import etree


class PrepareData:
    def __init__(self, file_name):
        """
        Load file from local storage
        @param file_name
        """
        self.file_name = file_name
        try:
            print
            file_name
            e = etree.ElementTree.parse("bank_train.xml")
            print
            e
            # in_file = open('data.csv', 'r')
            # self.db = MySQLdb.connect(
            #     user='root',
            #     db=adb,
            #     passwd='adminpass',
            #     host='localhost')
        except:
            print
            "An error has been raised while loading file."
