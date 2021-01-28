from ground_truths import ground_truths_politicians, ground_truths_pundits
from re import sub
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
from sklearn.decomposition import PCA
from nonsense_tester import returnNonsense


class VectorComparisonModel:
    def __init__(
        self,
        testuser=None,
        tfidf=True,
        ignore=tuple(),
        limit=0,
        sentiment=None,
        rts=True,
        n_comps=75,
    ):
        self.rts = rts
        self.sentiment = sentiment
        self.limit = limit
        self.ignore = ignore
        self.n_comps = n_comps

        if testuser is None:
            self.training_users = [
                x for x in ground_truths_politicians.keys() if x not in ignore
            ]
            self.test_users = [
                y for y in ground_truths_pundits.keys() if y not in ignore
            ]
            self.ytrain = np.array(
                [list(ground_truths_politicians[x]) for x in self.training_users]
            )
            self.ytest = np.array(
                [list(ground_truths_pundits[x]) for x in self.test_users]
            )
        else:
            self.test_users = [testuser]
            self.training_users = [
                x
                for x in ground_truths_politicians.keys()
                if x not in ignore + (testuser,)
            ]
            self.ytrain = np.array(
                [list(ground_truths_politicians[x]) for x in self.training_users]
            )
            try:
                self.ytest = np.array(
                    [list(ground_truths_politicians[x]) for x in self.test_users]
                )
            except KeyError:
                self.ytest = np.array(
                    [list(ground_truths_pundits[x]) for x in self.test_users]
                )

        if tfidf:
            self.vectorizer = TfidfVectorizer()
        else:
            self.vectorizer = CountVectorizer()

    def makedataframes(self):
        # training df
        training_users_tweets_dict = {
            x: get_tweets(x, self.rts, self.sentiment, self.limit, self.ignore)
            for x in self.training_users
            if x not in self.ignore
        }
        training_users_tweets_df = pd.DataFrame.from_dict(
            training_users_tweets_dict, orient="index", columns=["cleaned"]
        )

        # testing df
        test_users_tweets_dict = {
            x: get_tweets(x, self.rts, self.sentiment, self.limit, self.ignore)
            for x in self.test_users
            if x not in self.ignore
        }
        test_users_tweets_df = pd.DataFrame.from_dict(
            test_users_tweets_dict, orient="index", columns=["cleaned"]
        )

        return training_users_tweets_df, test_users_tweets_df

    def createvectors(
        self, training, testing, pca=True, topic_model=False, n_topics=8, lsi=False
    ):
        self.vectorizer.fit(testing["cleaned"])

        # make training vectors
        training["tfidf_vector"] = list(
            self.vectorizer.transform(training["cleaned"]).toarray()
        )
        training_vecs = np.vstack(
            tuple([x for x in training["tfidf_vector"].to_numpy()])
        )

        # make testing vectors
        athing = self.vectorizer.transform(testing["cleaned"])
        # print('Number of features (post-processing): {}\n'.format(athing.shape[1]))  # optional
        testing["tfidf_vector"] = list(athing.toarray())
        testing_vecs = np.vstack(tuple([x for x in testing["tfidf_vector"].to_numpy()]))

        if pca:
            reducer = PCA(n_components=self.n_comps)
            training_vecs = reducer.fit_transform(training_vecs)
            testing_vecs = reducer.transform(testing_vecs)

        if topic_model:
            if lsi:
                from sklearn.decomposition import TruncatedSVD

                model = TruncatedSVD(n_components=n_topics)
            else:
                from sklearn.decomposition import LatentDirichletAllocation

                model = LatentDirichletAllocation(n_components=n_topics)

            training_vecs = model.fit_transform(training_vecs)
            testing_vecs = model.transform(testing_vecs)

        return training_vecs, testing_vecs

    def runregression(self, xtrain, xtest):
        regr = linear_model.LinearRegression()
        regr.fit(xtrain, self.ytrain)
        ypred = regr.predict(xtest)

        return ypred


def get_tweets(handle, rts=True, sentiment=None, limit=0, exclude=None):
    connection = sqlite3.connect("tweet_data.db")

    sql = "select cleaned from tweet_text where author_handle = '{}'".format(handle)
    if not rts:
        sql += " and tweet not like '%RT%'"
    if sentiment:
        sql += " and sentiment = '{}'".format(sentiment)
    if exclude:
        sql += "and author_handle not in {}".format(str(exclude))
    if limit != 0:
        sql += " limit {}".format(limit)

    all_tweets = pd.read_sql_query(sql, connection)

    connection.close()
    return sub("\n", "", " ".join(all_tweets["cleaned"].tolist()))


def main(user=None, print_results=True, rec=False, comps=0, save=False):
    # run model
    model = VectorComparisonModel(testuser=user, n_comps=comps)
    train_df, test_df = model.makedataframes()
    train_vecs, test_vecs = model.createvectors(train_df, test_df)
    ypred = pd.DataFrame(model.runregression(train_vecs, test_vecs))

    # get model error as a vector distance
    dist = ((ypred - model.ytest) ** 2).sum(axis=1) ** 0.5

    final = pd.concat(
        [pd.Series([x for x in model.test_users]), pd.Series(dist)], axis=1
    )

    final = pd.concat([final, ypred], axis=1, sort=False)

    if print_results:
        print("Regression results:")
    pd.set_option("display.max_columns", 5)

    final.columns = [
        "author_handle",
        "dist",
        "social_score_estimate",
        "economic_score_estimate",
    ]

    nonsense = []
    for author, error, s_score, e_score in final.itertuples(index=False):
        nonsense.append(returnNonsense(author, s_score, e_score))

    failed = pd.Series(nonsense)

    final = pd.concat([final, failed], axis=1, sort=False)

    if user is None:
        from plot_results import plotrunNolans2

        final.index = final.author_handle
        final.columns = [
            "User",
            "Model Error",
            "Social Score Estimate",
            "Economic Score Estimate",
            "Failed",
        ]
        if print_results:
            print(final[["Economic Score Estimate", "Social Score Estimate"]])

        if rec:
            return final
        else:
            final.columns = ["author_handle", "dist", "s_score", "e_score", "Failed"]
            if save:
                conn = sqlite3.connect("tweet_data.db")
                curs = conn.cursor()
                sql = """insert into results2 (id, author_handle, model_error, s_score, e_score, nonsense)
                         VALUES (?,?,?,?,?,?);"""
                for author, dist, s_score, e_score, failed in final.values:
                    curs.execute(sql, (save, author, dist, s_score, e_score, failed))
                conn.commit()
                conn.close()

            else:
                plotrunNolans2(final[["author_handle", "s_score", "e_score", "Failed"]])

    else:
        from plot_results import plotrunNolans

        vals = final.values[0]
        if print_results:
            print(
                "Test User: {}\nModel Error: {}\nEconomic Score Estimate: {}\nSocial Score Estimate: {}".format(
                    vals[0], vals[1], vals[3], vals[2]
                )
            )

        final.columns = ["author_handle", "model_error", "s_score", "e_score", "Failed"]
        if rec:
            return final
        else:
            plotrunNolans(final[["author_handle", "s_score", "e_score"]], user)


def getrecommendations(user):
    final = main(user, print_results=False, rec=True)

    # find most similiar training users via row-wise comparison
    final.columns = ["author_handle", "error", 0, 1, "nonsense"]
    final = final[[0, 1]]
    compare_with = pd.DataFrame.from_dict(ground_truths_politicians, orient="index")
    merged = final.append(compare_with)

    # get test distribution
    test_series = merged.iloc[0]

    # calculate loss and get rid of rows with a negative loss
    loss = []
    for person in merged.index[1:]:
        doc = merged.loc[person]
        dist = np.linalg.norm(test_series - doc)
        loss.append((person, dist))

    sorted_loss = sorted(loss, key=lambda x: x[1])
    best_loss = sorted_loss[: len(ground_truths_politicians.keys()) // 25]
    print(
        "Hi {}! Based on your results, we recommend that you check out these Twitter users:\n{}\n".format(
            user, ", ".join([x[0] for x in best_loss])
        )
    )


if __name__ == "__main__":
    main()
