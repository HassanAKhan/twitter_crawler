
import tweepy
import pymongo
import sys

# authorization tokens
consumer_key = "vOc9LolIXAJI2JW6X3u33GK7h"
consumer_secret = "Pz2FIUI3Oy3s0HhMG6YeBfZY6Gm52ebEEEO0t4n6rPrQYILvX9"
access_key = "728347342034894850-Y7WmUZmjBypTiyjhL96qYaTimsGYYnH"
access_secret = "yl709OhGtdGqfPGhl1SNNHcF0yCXVROFh8MruZ53duaPo"


class StreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)


    def on_error(self, status_code):
        print("Encountered streaming error (", status_code, ")")
        sys.exit()


if __name__ == "__main__":
    # complete authorization and initialize API endpoint
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize stream
    streamListener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=streamListener, languages=["en"])
    tags = ["Corona"]
    stream.filter(track=tags)