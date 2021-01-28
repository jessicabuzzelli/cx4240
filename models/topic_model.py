from ground_truths import ground_truths_politicians, ground_truths_pundits
from re import sub
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model


def get_tweets(handle, rts=True, sentiment=None, limit=False, exclude=None):
    connection = sqlite3.connect("tweet_data.db")

    sql = "select cleaned from tweet_text where author_handle = '{}'".format(handle)
    if rts is False:
        sql += " and tweet not like '%RT%'"
    if sentiment:
        sql += " and sentiment = '{}'".format(sentiment)
    if exclude:
        sql += "and author_handle not in {}".format(str(exclude))
    if limit:
        sql += " limit {}".format(limit)

    all_tweets = pd.read_sql_query(sql, connection)

    connection.close()
    return sub("\n", "", " ".join(all_tweets["cleaned"].tolist()))


class LDA:
    def __init__(self, tfidf=True, ignore=tuple(), limit=0, sentiment=None, rts=True):
        self.rts = rts
        self.sentiment = sentiment
        self.limit = limit
        self.exclude = ignore

        self.training_users = [
            x for x in ground_truths_politicians.keys() if x not in ignore
        ]
        self.test_users = [y for y in ground_truths_pundits.keys() if y not in ignore]
        self.ytrain = np.array(
            [list(ground_truths_politicians[x]) for x in self.training_users]
        )
        self.ytest = np.array([list(ground_truths_pundits[x]) for x in self.test_users])

        self.ignore = ignore

        if tfidf:
            self.vectorizer = TfidfVectorizer()
        else:
            self.vectorizer = CountVectorizer()

    def make_dfs(self):
        # training df
        training_users_tweets_dict = {
            x: get_tweets(x, self.rts, self.sentiment, self.limit, self.exclude)
            for x in self.training_users
            if x not in self.ignore
        }
        training_users_tweets_df = pd.DataFrame.from_dict(
            training_users_tweets_dict, orient="index", columns=["cleaned"]
        )

        # testing df
        test_users_tweets_dict = {
            x: get_tweets(x, self.rts, self.sentiment, self.limit, self.exclude)
            for x in self.test_users
            if x not in self.ignore
        }
        test_users_tweets_df = pd.DataFrame.from_dict(
            test_users_tweets_dict, orient="index", columns=["cleaned"]
        )

        return training_users_tweets_df, test_users_tweets_df

    def create_vecs(self, training, testing):
        training["tweets"] = training["cleaned"]
        testing["tweets"] = testing["cleaned"]

        # transform corpus
        self.vectorizer.fit(testing["tweets"])

        # create training vectors
        training["tfidf_vector"] = list(
            self.vectorizer.transform(training["tweets"]).toarray()
        )
        training_vecs = np.vstack(
            tuple([x for x in training["tfidf_vector"].to_numpy()])
        )

        # create testing vectors
        athing = self.vectorizer.transform(testing["tweets"])
        print("Number of features (post-processing): {}\n".format(athing.shape[1]))
        testing["tfidf_vector"] = list(athing.toarray())
        testing_vecs = np.vstack(tuple([x for x in testing["tfidf_vector"].to_numpy()]))

        return training_vecs, testing_vecs

    def run_LDA(self, training_vecs, testing_vecs, num_topics=6, lsi=False):
        self.num_topics = num_topics
        if lsi:
            from sklearn.decomposition import TruncatedSVD

            model = TruncatedSVD(n_components=num_topics)
        else:
            from sklearn.decomposition import LatentDirichletAllocation

            model = LatentDirichletAllocation(n_components=num_topics)

        xtrain = model.fit_transform(training_vecs)
        xtest = model.transform(testing_vecs)

        return xtrain, xtest

    def run_regression(self, xtrain, xtest):
        regr = linear_model.LinearRegression()
        regr.fit(xtrain, self.ytrain)
        ypred = regr.predict(xtest)

        return ypred


def main(num=6):
    # run model
    model = LDA(tfidf=True)
    train_df, test_df = model.make_dfs()
    train_vecs, test_vecs = model.create_vecs(train_df, test_df)
    xtrain, xtest = model.run_LDA(train_vecs, test_vecs, num_topics=num)
    ypred = pd.DataFrame(model.run_regression(xtrain, xtest))

    # get model error as vector distance
    dist = ((ypred - model.ytest) ** 2).sum(axis=1) ** 0.5

    final = pd.concat(
        [pd.Series([x for x in model.test_users]), pd.Series(dist)], axis=1
    )

    final = pd.concat([final, ypred], axis=1, sort=False)

    nonsense = final[final["dist"] > 0.3]["author_handle"].values

    final = pd.concat([final, nonsense], axis=1, sort=False)
    final.columns = [
        "author_handle",
        "model_error",
        "social_score_estimate",
        "economic_score_estimate",
        "violated_user_bounds",
    ]

    print("Number of components used in LDA: {} \n".format(model.num_topics))
    print("Regression results:")
    pd.set_option("display.max_columns", 4)
    final.set_index("author_handle", drop=True)
    print(
        final[
            ["economic_score_estimate", "social_score_estimate", "violated_user_bounds"]
        ]
    )


if __name__ == "__main__":
    main()
