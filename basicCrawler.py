
import tweepy
import pymongo
import sys

# authorization tokens
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""


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