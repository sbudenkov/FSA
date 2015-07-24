import sys
import tweepy
from tweepy import OAuthHandler
from twitter_keys import consumer_key, consumer_secret, access_token, access_secret

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    print(status.text.encode(sys.stdout.encoding, errors='replace'))

# for status in tweepy.Cursor(api.home_timeline).items(10):
#     # Process a single status
#     process_or_store(status.json)
#
# for friend in tweepy.Cursor(api.friends).items():
#     process_or_store(friend.json)
#
# for tweet in tweepy.Cursor(api.user_timeline).items():
#     process_or_store(tweet.json)