# -*- coding: utf-8 -*-

from tweepy import Stream
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from twitter_keys import consumer_key, consumer_secret, access_token, access_secret
 
class MyListener(StreamListener):
    def on_status(self, status):
        print status.text

    def on_data(self, data):
        try:
            print("Data ...")
            with open('test.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)
file = open('test.json', 'wb')
file.close()
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=[u'#казань2015'])