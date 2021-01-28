import tweepy
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ground_truths import ground_truths_politicians, ground_truths_pundits
import api_keys


class TwitterHandler(object):
    def __init__(self):
        __access_token__ = api_keys.access_token
        __access_secret__ = api_keys.access_secret
        __consumer_key__ = api_keys.consumer_key
        __consumer_secret__ = api_keys.consumer_secret

        self.analyser = SentimentIntensityAnalyzer()

        try:
            auth = tweepy.OAuthHandler(__consumer_key__, __consumer_secret__)
            auth.set_access_token(__access_token__, __access_secret__)
            self.api = tweepy.API(
                auth,
                wait_on_rate_limit=True,
                wait_on_rate_limit_notify=True,
                compression=False,
            )

        except tweepy.TweepError:
            print("Error: Authentication Failed")

        self.conn = sqlite3.connect("tweet_data.db")
        self.cursor = self.conn.cursor()

        # ensure tweet table exists
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS tweet_text 
        (id TEXT PRIMARY KEY, author_handle TEXT, tweet TEXT, sentiment TEXT, pos_score REAL, neg_score REAL,
        neu_score REAL, compound_score REAL);"""
        )

        # ensure user table exists
        # cursor.execute('''DROP TABLE IF EXISTS user;''') # if need to refresh user stats
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS user 
        (author_handle TEXT PRIMARY KEY, follower_count INTEGER, description TEXT, user_id INTEGER,
        name TEXT, type TEXT);"""
        )

        self.conn.commit()

    def getsentiment(self, tweet):
        # strip excess whitespace, this step might be unnecessary
        tweet = tweet.replace("\n", " ")
        tweet = tweet.replace("\t", " ")

        # get polarity scores from Vader
        score = self.analyser.polarity_scores(tweet)

        # Assign presiding sentiment
        if score["compound"] > 0.05:
            sentiment = "positive"
        elif score["compound"] < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return sentiment, score["pos"], score["neg"], score["neu"], score["compound"]

    def gettweets(self, query, count):
        tweets = []

        try:
            search_results = []

            # make initial request for most recent tweets (maximum allowed count = 200)
            new_tweets = self.api.user_timeline(
                screen_name=query, count=count, tweet_mode="extended"
            )  # tweets are truncated to 180 chars by default
            search_results.extend(new_tweets)

            # save id of the youngest tweet -- this will become oldest tweet in while loop
            oldest = search_results[-1].id - 1

            while (
                len(new_tweets) > 0
            ):  # keep pulling in tweets in increments of 200 until none remain
                new_tweets = self.api.user_timeline(
                    screen_name=query, count=200, max_id=oldest, tweet_mode="extended"
                )

                # add new tweets and reset max_id value
                search_results.extend(new_tweets)
                oldest = search_results[-1].id - 1

            # get user metadata from a tweet
            atweet = search_results[0]
            user_data = {
                "author_handle": atweet.user.screen_name,
                "follower_count": atweet.user.followers_count,
                "description": atweet.user.description,
                "user_id": atweet.user.id,
                "name": atweet.user.name,
            }

            for tweet in search_results:
                parsed_tweet = {}

                try:
                    parsed_tweet["text"] = tweet.full_text
                except AttributeError:
                    parsed_tweet["text"] = tweet.text

                # assign sentiment polarity scores
                (
                    parsed_tweet["sentiment"],
                    parsed_tweet["pos"],
                    parsed_tweet["neg"],
                    parsed_tweet["neu"],
                    parsed_tweet["compound"],
                ) = self.getsentiment(parsed_tweet["text"])

                parsed_tweet["date"] = tweet.created_at
                parsed_tweet["id"] = tweet.id

                tweets.append(parsed_tweet)

            # return parsed tweets
            return tweets, user_data

        except IndexError:
            print("{} has no tweets available.".format(query[5:]))
            return None, None

        except tweepy.TweepError as e:
            print("Error : " + str(e))
            return None, None

    def exportresults(self, tweets, user_data, handle):
        if handle in ground_truths_politicians.keys():
            user_type = "politician"
        else:
            user_type = "pundit"

        try:
            user_attrs = (
                handle,
                user_data["follower_count"],
                user_data["description"],
                user_data["user_id"],
                user_data["name"],
                user_type,
            )
            try:
                self.cursor.execute(
                    """INSERT INTO user 
                                    (author_handle, follower_count, description, user_id, name, type) 
                                     VALUES (?, ?, ?, ?, ?, ?);""",
                    user_attrs,
                )
            except sqlite3.IntegrityError:
                pass

        except TypeError:
            pass

        if tweets is []:
            return

        # write new tweets to db
        for tweet in tweets:
            if tweet is None:
                pass

            tweet_attrs = (
                tweet["id"],
                handle,
                tweet["text"],
                tweet["sentiment"],
                tweet["pos"],
                tweet["neg"],
                tweet["neu"],
                tweet["compound"],
            )

            try:
                self.cursor.execute(
                    """INSERT INTO tweet_text 
                                    (id,
                                     author_handle,
                                     tweet,
                                     sentiment,
                                     pos_score, 
                                     neg_score, 
                                     neu_score, 
                                     compound_score) VALUES 
                                     (?, ?, ?, ?, ?, ?, ?, ?);""",
                    tweet_attrs,
                )

            except sqlite3.IntegrityError:
                pass

        self.conn.commit()


if __name__ == "__main__":
    training_list = list(ground_truths_politicians.keys()) + list(
        ground_truths_pundits.keys()
    )

    searcher = TwitterHandler()

    for handle in ground_truths_politicians.keys():
        tweets, user_data = searcher.gettweets(query=handle, count=200)
        searcher.exportresults(tweets, user_data, handle)
        print("finished {}".format(handle))

    searcher.conn.close()
