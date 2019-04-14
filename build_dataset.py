import tweepy
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ground_truths import ground_truths_politicians, ground_truths_pundits
import api_keys

class TwitterHandler(object):

    def __init__(self):
        # rate limit: 15 calls per 15 minute window
        ACCESS_TOKEN = api_keys.access_token
        ACCESS_SECRET = api_keys.access_secret
        CONSUMER_KEY = api_keys.consumer_key
        CONSUMER_SECRET = api_keys.consumer_secret

        self.analyser = SentimentIntensityAnalyzer()

        try:
            auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
            auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
            self.api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True, compression = False)

        except:
            print("Error: Authentication Failed")

        self.conn = sqlite3.connect('tweet_data.db')
        self.cursor = self.conn.cursor()

        # create tweet table if not exists
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS tweet_text 
        (id TEXT PRIMARY KEY, author_handle TEXT, tweet TEXT, sentiment TEXT, pos_score REAL, neg_score REAL,
        neu_score REAL, compound_score REAL);''')


        # create user table if not exists
        # cursor.execute('''DROP TABLE IF EXISTS user;''') # if need to refresh user stats
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS user 
        (author_handle TEXT PRIMARY KEY, follower_count INTEGER, description TEXT, user_id INTEGER,
        name TEXT, type TEXT);''')

        self.conn.commit()

    def getSentiment(self, tweet):
        tweet = tweet.replace('\n',' ')
        tweet = tweet.replace('\t',' ')

        # polarity score of the tweet = weighted average of polarity of comprising sentences

        score = self.analyser.polarity_scores(tweet)

        # Assign presiding sentiment
        if score['compound'] > 0.05:
            sentiment = 'positive'
        elif score['compound'] < -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return sentiment, score['pos'], score['neg'], score['neu'], score['compound']

    def getTweets(self, query, count):
        tweets = []

        try:
            search_results = []

            # make initial request for most recent tweets (200 is the maximum allowed count)
            new_tweets = self.api.user_timeline(screen_name=query[5:],
                                                count=count,
                                                tweet_mode='extended')

            # save most recent tweets
            search_results.extend(new_tweets)

            # save the id of the oldest tweet less one
            oldest = search_results[-1].id - 1

            while len(new_tweets) > 0:
                # all subsequent requests use the max_id param to prevent duplicates
                new_tweets = self.api.user_timeline(screen_name=query[5:],
                                                    count=200,
                                                    max_id=oldest,
                                                    tweet_mode='extended')

                # save most recent tweets
                search_results.extend(new_tweets)

                # update the id of the oldest tweet less one
                oldest = search_results[-1].id - 1

            #search_results = self.api.search(q=query, tweet_mode = 'extended', count=count) # operates via searching for 100 most recent tweets matching "from:handle", does NOT truncate tweets
            #search_results = self.api.user_timeline(id=query[5:])
            # format user data from info retrieved with the first tweet of the query
            atweet = search_results[0]
            user_data = {'author_handle': atweet.user.screen_name, 'follower_count': atweet.user.followers_count, 'description': atweet.user.description, 'user_id': atweet.user.id, 'name': atweet.user.name}

            for tweet in search_results:
                parsed_tweet = {}

                try:
                    parsed_tweet['text'] = tweet.full_text
                except AttributeError:
                    parsed_tweet['text'] = tweet.text

                parsed_tweet['sentiment'], parsed_tweet['pos'], parsed_tweet['neg'], parsed_tweet['neu'], parsed_tweet['compound'] = self.getSentiment(parsed_tweet['text'])
                parsed_tweet['date'] = tweet.created_at
                parsed_tweet['id'] = tweet.id

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            # return parsed tweets
            return tweets, user_data

        except IndexError:
            print('{} has no tweets available.'.format(query[5:]))
            return None, None

        except tweepy.TweepError as e:
            print("Error : " + str(e))
            return None, None

    def exportResults(self, tweets, user_data, handle):
        if handle in ground_truths_politicians.keys():
            user_type = 'politician'
        else:
            user_type = 'pundit'

        try:
            user_attrs = (handle,
                          user_data['follower_count'],
                          user_data['description'],
                          user_data['user_id'],
                          user_data['name'], user_type)
            try:
                self.cursor.execute('''INSERT INTO user 
                                    (author_handle, follower_count, description, user_id, name, type) 
                                     VALUES (?, ?, ?, ?, ?, ?);''', user_attrs)
            except sqlite3.IntegrityError:
                pass

        except TypeError:
        # write new users to db
            pass

        if tweets == []:
            return

        # write new tweets to db
        for tweet in tweets:
            if tweet == None:
                pass

            tweet_attrs = (tweet['id'],
                           handle,
                           tweet['text'],
                           tweet['sentiment'],
                           tweet['pos'],
                           tweet['neg'],
                           tweet['neu'],
                           tweet['compound'])

            try:
                self.cursor.execute('''INSERT INTO tweet_text 
                                    (id,
                                     author_handle,
                                     tweet,
                                     sentiment,
                                     pos_score, 
                                     neg_score, 
                                     neu_score, 
                                     compound_score) VALUES 
                                     (?, ?, ?, ?, ?, ?, ?, ?);''',
                                     tweet_attrs)

            except sqlite3.IntegrityError:
                pass

        self.conn.commit()

if __name__ == "__main__":
    training_list = list(ground_truths_politicians.keys()) + list(ground_truths_pundits.keys())

    searcher = TwitterHandler()

    for handle in ['patilaakanksha']:
        tweets, user_data = searcher.getTweets(query='from:{}'.format(handle), count=200)
        searcher.exportResults(tweets, user_data, handle)
        print('finished {}'.format(handle))

    searcher.conn.close()