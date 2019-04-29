from ground_truths import ground_truths_politicians, ground_truths_pundits
from re import sub
import sqlite3
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim.corpora
from gensim.models import TfidfModel, LdaModel
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class LDA:
    def __init__(self, cv_id):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')
        self.cv_id = cv_id

    def preprocess(self, text):
        dirty_text = [r'http\S+', r'@\S+', r'#', r'&amp;']

        # manually add noisy words as they appear in topic composition results
        ignore = ['thing', 'called', 'say', 'good',
                  'said', 'go', 'today', 'day', 'want', 'need',
                  'hear', 'get', 'like', 'come', 'week', 'think',
                  'thought', 'month', 'tonight', 'year', 'hear', 'need',
                  'know', 'talk', 'RT']

        processed_ignores = []

        for word in ignore + [x for x in ground_truths_pundits.keys()] + [y for y in ground_truths_politicians.keys()]:
            processed_ignores.append(self.stemmer.stem(self.lemmatizer.lemmatize(word)))

        for pattern in dirty_text:
            text = sub(pattern, '', text)

        lemmed_and_stemmed = []

        for token in simple_preprocess(text):
            if token not in STOPWORDS and len(token) > 3 and token not in processed_ignores:
                lemmed_and_stemmed.append(self.stemmer.stem(self.lemmatizer.lemmatize(token)))

        return lemmed_and_stemmed

    def get_corpus(self, rts=True, ignore=(), sentiment=None, limit=0, tfidf=True):
        self.ignore = ignore
        self.limit = limit
        self.sentiment = sentiment
        self.rts = rts

        self.users = [x for x in ground_truths_politicians.keys() if x not in ignore]

        # make dataframe for model training
        connection = sqlite3.connect('tweet_data.db')
        sql = "select id, tweet, author_handle, compound_score from tweet_text where author_handle in {}"\
            .format(str(tuple(self.users)))

        if sentiment is True:
            sql += " and sentiment = '{}'".format(sentiment)
        if rts is False:
            sql+= " and tweet not like '%RT%'"
        if limit is True:
            sql += ' limit {}'.format(limit)

        df = pd.read_sql_query(sql, connection, index_col='id')
        connection.close()

        # process tweets, make dictionary, encode corpus
        df['tweet'] = df['tweet'].map(self.preprocess)  # tokenize + lemmatize + stem

        self.dictionary = gensim.corpora.Dictionary(df['tweet'])
        corpus = [self.dictionary.doc2bow(doc) for doc in df['tweet']]
        if tfidf is True:
            tfidf_corpus = TfidfModel(corpus)
            corpus = tfidf_corpus[corpus]

        return corpus, df

    def getScores(self, corpus, df, num_topics=10):
        self.num_topics = num_topics

        # form LDA model
        self.LDA = LdaModel(corpus=corpus, id2word=self.dictionary, num_topics=num_topics,
                                              update_every=1, passes=2)

        # test each training user against the LDA model and get their normalized topic polarity scores
        user_sentiments_by_topic = {}
        for user in self.users:
            user_df = df[df['author_handle'] == user]
            user_vector = [self.dictionary.doc2bow(tweet) for tweet in user_df['tweet']]
            best_score_df = pd.DataFrame([sorted(self.LDA[vec], key=lambda tup: -1*tup[1])[0] for vec in user_vector],
                                         index=user_df.index)
            user_df.loc[:, 'best_topic'] = best_score_df.loc[:, 0]
            user_df.loc[:, 'best_score'] = best_score_df.loc[:, 1]
            topic_sentiment_scores = [(
                                       user_df[user_df['best_topic'] == topic]['compound_score'] *
                                       user_df[user_df['best_topic'] == topic].loc[:, 'best_score'] /
                                       user_df[user_df['best_topic'] == topic].loc[:, 'best_score'].sum(axis=0)
                                       ).sum(axis=0)
                                      for topic in range(num_topics)]
            user_sentiments_by_topic[user] = topic_sentiment_scores

        return user_sentiments_by_topic

    def get_nolan_scores(self, testuser, scores, div_number):
        # form testuser's array
        connection = sqlite3.connect('tweet_data.db')
        sql = "select id, tweet, author_handle, compound_score from tweet_text where author_handle = '{}'".format(
            testuser)
        if self.sentiment is True:
            sql += " and sentiment = '{}'".format(self.sentiment)
        if self.rts is False:
            sql+= " and tweet not like '%RT%'"
        if self.limit is True:
            sql += ' limit {}'.format(self.limit)
        test_df = pd.read_sql_query(sql, connection, index_col='id')
        connection.close()

        # get normalized topic polarity scores for the testuser from LDA
        test_df['tweet'] = test_df['tweet'].map(self.preprocess)  # tokenize + lemmatize + stem
        user_vector = [self.dictionary.doc2bow(tweet) for tweet in test_df['tweet']]
        best_score_df = pd.DataFrame([sorted(self.LDA[vec], key=lambda tup: tup[1])[0] for vec in user_vector],
                                     index=test_df.index)
        test_df.loc[:, 'best_topic'] = best_score_df.loc[:, 0]
        test_df.loc[:, 'best_score'] = best_score_df.loc[:, 1]
        test_scores = [(
                        test_df[test_df['best_topic'] == topic]['compound_score'] *
                        test_df[test_df['best_topic'] == topic].loc[:, 'best_score'] /
                        test_df[test_df['best_topic'] == topic].loc[:, 'best_score'].sum(axis=0)
                        ).sum(axis=0)
                       for topic in range(self.num_topics)]
        test_array = pd.DataFrame(np.array(test_scores), columns=[testuser]).T

        # use DFs for getting distances between vectors of normalized topic scores (model errors)
        compare_with = pd.DataFrame.from_dict(scores, orient='index')
        merged = test_array.append(compare_with)

        # get test distribution
        test_series = merged.iloc[0]

        # calculate differences between training users and the test user
        loss = []
        for person in merged.index[1:]:
            doc = merged.loc[person]
            dist = np.linalg.norm(test_series - doc)
            loss.append((person, dist))

        sorted_loss = sorted(loss, key=lambda x: x[1])
        best_loss = sorted_loss[:len(self.users) // div_number]

        # below: scores from closest results in the test set
        # this model explores using top matches to determine a testuser's score from their weighted average
        # in other models, score estimations came from linear regression; ideally, this would also use regression

        scores = np.array([x[1] for x in best_loss])
        if np.absolute(scores).sum() != 0:
            normalized_error = np.absolute(scores) / np.absolute(scores).sum()
        else:
            normalized_error = [0] * len(best_loss)

        s_score = 0
        e_score = 0
        for idx in range(len(best_loss)):
            s, e = ground_truths_politicians[best_loss[idx][0]]
            s_score += s * normalized_error[idx]
            e_score += e * normalized_error[idx]

        s_true, e_true = ground_truths_pundits[testuser]
        error = (((s_true - s_score) ** 2) + ((e_true - e_score) ** 2)) ** .5

        return [testuser, self.cv_id, s_score, e_score, error]


def batch_test(cv_id, sentiment, limit, ignore, rts, num_topics, tfidf, blind, div_number):
    from time import time
    start = time()

    results = []
    model = LDA(cv_id=cv_id)

    if blind is False:
        corpus, df = model.get_corpus(rts=rts, ignore=ignore, sentiment=sentiment, limit=limit, tfidf=tfidf)
        master_scores = model.getScores(corpus, df, num_topics=num_topics)

        for user in ground_truths_pundits.keys():
            scores = master_scores.copy()
            nolan_scores = model.get_nolan_scores(div_number=div_number, testuser=user, scores=scores)
            results.append(nolan_scores)

        end = time()
        print('Cross validation took this long: {}'.format(end-start))

        print(results)
        return results

    else:
        for user in ground_truths_pundits.keys():
            corpus, df = model.get_corpus(rts=rts, ignore=ignore, sentiment=sentiment, limit=limit, tfidf=tfidf)
            scores_dict = model.getScores(corpus, df, num_topics=num_topics)
            nolan_scores = model.get_nolan_scores(testuser=user, scores=scores_dict, div_number=div_number)
            results.append(nolan_scores)

        end = time()
        print('Cross validation took this long: {}'.format(end - start))

        print(results)
        return results


if __name__ == '__main__':
    testing = True
    combos = [[0, 0, 'LDA', False, True, True, 0, None, 8, tuple(), 15]]  # example

    for cv_id, run_id, model, blind, tfidf, rts, user_limit, sentiment, num_topics, ignored_users, div_number in combos:
        results = batch_test(cv_id, sentiment, user_limit, ignored_users, rts, num_topics, tfidf, blind, div_number)

        if testing == False:
            conn = sqlite3.connect('tweet_data.db')
            cursor = conn.cursor()

            cursor.executemany(master_combos.insert_results_sql, results)

            cursor.execute(master_combos.insert_runs_sql,
                           (cv_id,
                            run_id,
                            model,
                            str(blind).capitalize(),
                            str(tfidf).capitalize(),
                            str(rts).capitalize(),
                            user_limit,
                            sentiment,
                            num_topics,
                            '+'.join(list(ignored_users)),
                            div_number
                            )
                           )
            conn.commit()
            conn.close()
