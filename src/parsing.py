__author__ = 'budenkovss'
# coding: utf8

import csv as csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.decomposition import PCA
import coding

import feedparser
import pickle
import re
import nltk

coding.setup_console("cp1251")


in_file = open('data.csv', 'r')      # пусть файл у нас записан в cp1251
text = in_file.read()                # читаем из строку байтов
unicode_text = text.decode('cp1251') # раскодируем текст из cp1251.
#print unicode_text

csv_file_object = csv.reader(unicode_text)       #open('data.csv', 'rb'), quoting=csv.QUOTE_MINIMAL, delimiter=';')
#header = csv_file_object.next()
data=[]

for row in csv_file_object:
    data.append(row)

data = np.array(data)
data1 = data[:150] # закодируем текст в виндовую кодировку.

out_file = open('out.txt',"wb")
out_file.write(str(data1).encode('cp1251'))
out_file.close()

print nltk.download()


class CaptureFeeds:

    def __init__(self):
        for (i, url) in enumerate(self.rss_feeds_list()):
            self.capture_as_pickled_feed(url.strip(), i)

    def rss_feeds_list(self):
        f = open('feeds_list.txt', 'r')
        list = f.readlines()
        f.close
        return list

    def capture_as_pickled_feed(self, url, feed_index):
        feed = feedparser.parse(url)
        f = open('data/feed_' + str(feed_index) + '.pkl', 'w')
        pickle.dump(feed, f)
        f.close()

if __name__ == "__main__":
    cf = CaptureFeeds()

class RssItem:

    #regex = re.compile('[%s]' % re.escape(string.punctuation))

    def normalized_words(self, article_text):
        words = []
        oneline = article_text.replace('\n', ' ')
        cleaned = nltk.clean_html(oneline.strip())
        toks1   = cleaned.split()
        for t1 in toks1:
            translated = self.regex.sub('', t1)
            toks2 = translated.split()
            for t2 in toks2:
                t2s = t2.strip().lower()
                if self.stop_words.has_key(t2s):
                    pass
                else:
                    words.append(t2s)
        return words


    def features(self, top_words):
        word_set = set(self.all_words)
        features = {}
        for w in top_words:
            features["w_%s" % w] = (w in word_set)
        return features

    # def classify_reuters(self):
    #     training_set = []
    #     for item in rss_items:
    #         features = item.features(top_words)
    #         tup = (features, item.category)  # tup is a 2-element tuple
    #         featuresets.append(tup)
    #     classifier = nltk.NaiveBayesClassifier.train(training_set)

      # for item in rss_items_to_classify:
      # features = item.features(top_words)
      # category = classifier.classify(feat)

def collect_all_words(self, items):
  all_words = []
  for item in items:
      for w in item.all_words:
          all_words.append(w)
  return all_words

def identify_top_words(self, all_words):
  freq_dist = nltk.FreqDist(w.lower() for w in all_words)
  return freq_dist.keys()[:1000]