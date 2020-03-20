
import tweepy
import pymongo
import datetime
# authorization tokens
consumer_key = "vOc9LolIXAJI2JW6X3u33GK7h"
consumer_secret = "Pz2FIUI3Oy3s0HhMG6YeBfZY6Gm52ebEEEO0t4n6rPrQYILvX9"
access_key = "728347342034894850-Y7WmUZmjBypTiyjhL96qYaTimsGYYnH"
access_secret = "yl709OhGtdGqfPGhl1SNNHcF0yCXVROFh8MruZ53duaPo"


class StreamListener(tweepy.StreamListener):
    def on_status(self, status):



        with open('stream_out.csv', "a", encoding='utf-8') as f:


            date = status.created_at
            tweet_id = status.id_str
            text = status.text
            special_char = [',','"',"'",'\n']

            for elm in special_char:
                text = text.replace(elm , '')

            text = text.strip('\n')
            user_id = status.user.id
            user_name = status.user.screen_name

            f.write("%s,%s,%s,%s,%s\n" % (str(date),str(tweet_id), str(text.rstrip()), str(user_id), str(user_name)))
    print('done')



if __name__ == "__main__":
    # complete authorization and initialize API endpoint
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize stream
    streamListener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=streamListener, languages=["en"])
    tags = ["Corona", "qurantine", "coronavirus", "virus", "covid", "stocks"]


    with open("stream_out.csv", "w", encoding='utf-8') as f:
        f.write("date,tweet_ID,text,user_id,user_name\n")
    stream.filter(track=tags)
    # stream.sample(languages=['en'])