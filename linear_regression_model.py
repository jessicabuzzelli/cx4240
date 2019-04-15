from ground_truths import ground_truths_politicians, ground_truths_pundits
from re import sub
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
from sklearn.decomposition import PCA
from nonsense_tester import returnNonsense

def getTweets(handle, RTs=True, sentiment=None, limit=0, exclude=None):
    connection = sqlite3.connect('tweet_data.db')

    sql = "select cleaned from tweet_text where author_handle = '{}'".format(handle)
    if RTs == False:
        sql += " and tweet not like '%RT%'"
    if sentiment != None:
        sql += " and sentiment = '{}'".format(sentiment)
    if exclude != None:
        sql += "and author_handle not in {}".format(str(exclude))
    if limit != 0:
        sql += " limit {}".format(limit)

    all_tweets = pd.read_sql_query(sql, connection)

    connection.close()
    return sub('\n', '', ' '.join(all_tweets['cleaned'].tolist()))

class VectorComparisonModel:
    def __init__(self, testuser=None, TFIDF=True, ignore=tuple(), limit=0, sentiment=None, RTs=True, n_comps=0):
        self.rts = RTs
        self.sentiment = sentiment
        self.limit = limit
        self.ignore = ignore
        self.n_comps = n_comps

        if testuser == None:
            self.training_users = [x for x in ground_truths_politicians.keys() if x not in ignore]
            self.test_users = [y for y in ground_truths_pundits.keys() if y not in ignore]
            self.ytrain = np.array([list(ground_truths_politicians[x]) for x in self.training_users])
            self.ytest = np.array([list(ground_truths_pundits[x]) for x in self.test_users])
        else:
            self.test_users = [testuser]
            self.training_users = [x for x in ground_truths_politicians.keys() if x not in ignore and x != testuser]
            self.ytrain = np.array([list(ground_truths_politicians[x]) for x in self.training_users])
            try:
                self.ytest = np.array([list(ground_truths_politicians[x]) for x in self.test_users])
            except KeyError:
                self.ytest = np.array([list(ground_truths_pundits[x]) for x in self.test_users])

        if TFIDF == True:
            self.vectorizer = TfidfVectorizer()
        else:
            self.vectorizer = CountVectorizer()

    def makeDataFrames(self):
        # training df
        training_users_tweets_dict = {x: getTweets(x, self.rts, self.sentiment, self.limit, self.ignore)
                                      for x in self.training_users if x not in self.ignore}
        training_users_tweets_df = pd.DataFrame.from_dict(training_users_tweets_dict, orient='index',
                                                          columns=['cleaned'])

        # testing df
        test_users_tweets_dict = {x: getTweets(x, self.rts, self.sentiment, self.limit, self.ignore)
                                  for x in self.test_users if x not in self.ignore}
        test_users_tweets_df = pd.DataFrame.from_dict(test_users_tweets_dict, orient='index', columns=['cleaned'])

        return training_users_tweets_df, test_users_tweets_df

    def createVectors(self, training, testing, pca=True):
        training['tweets'] = training['cleaned']
        testing['tweets'] = testing['cleaned']

        # fit vectorizer
        # self.vectorizer.fit(training['tweets'].append(testing['tweets']))
        self.vectorizer.fit(testing['tweets'])

        # make training vectors
        training['tfidf_vector'] = list(self.vectorizer.transform(training['tweets']).toarray())
        training_vecs = np.vstack(tuple([x for x in training['tfidf_vector'].to_numpy()]))

        # make testing vectors
        athing = self.vectorizer.transform(testing['tweets'])
        # print('Number of features (post-processing): {}\n'.format(athing.shape[1]))
        testing['tfidf_vector'] = list(athing.toarray())
        testing_vecs = np.vstack(tuple([x for x in testing['tfidf_vector'].to_numpy()]))

        if pca == True:
            reducer = PCA(n_components=self.n_comps)
            training_vecs = reducer.fit_transform(training_vecs)
            testing_vecs = reducer.transform(testing_vecs)

        return training_vecs, testing_vecs

    def runRegression(self, xtrain, xtest):
        regr = linear_model.LinearRegression()
        regr.fit(xtrain, self.ytrain)
        ypred = regr.predict(xtest)

        return ypred

def main(user=None, print_results=True, rec=False, comps=0, save=False):
    model = VectorComparisonModel(testuser=user, n_comps=comps)   # todo - get rid of this
    train_df, test_df = model.makeDataFrames()
    train_vecs, test_vecs = model.createVectors(train_df, test_df)
    ypred = pd.DataFrame(model.runRegression(train_vecs, test_vecs))

    dist = ((ypred - model.ytest) ** 2).sum(axis=1) ** .5

    final = pd.concat([pd.Series([x for x in model.test_users]),
                       pd.Series(dist)], axis=1)

    final = pd.concat([final, ypred], axis=1, sort=False)

    if print_results == True:
        print('Regression results:')
    pd.set_option('display.max_columns', 5)

    final.columns = ['author_handle', 'dist',
                     'social_score_estimate', 'economic_score_estimate']

    nonsense2 = []
    for author, error, s_score, e_score in final.itertuples(index=False):
        nonsense2.append(returnNonsense(author, s_score, e_score))

    failed = pd.Series(nonsense2)

    final = pd.concat([final, failed], axis=1, sort=False)
    #nonsense = [x[0] for x in final[final['Failed'] == True].values]

    if user == None:
        from plot_results import plotrunNolans2
        #final = final.set_index('author_handle', drop=True)
        final.index = final.author_handle
        final.columns = ['User','Model Error','Social Score Estimate', 'Economic Score Estimate','Failed']
        if print_results == True:
            print(final[['Economic Score Estimate', 'Social Score Estimate']])

        if rec == True:
            return final
        else:
            # print('Users who failed to fall within 0.3 units from our personal estimates are shown below:')
            final.columns = ['author_handle','dist','s_score','e_score','Failed']
            if save != False:
                conn = sqlite3.connect('tweet_data.db')
                curs = conn.cursor()
                sql = """insert into results2 (id, author_handle, model_error, s_score, e_score, nonsense) VALUES (?,?,?,?,?,?);"""
                for author, dist,s_score,e_score,failed in final.values:
                    # print([author,dist,s_score,e_score,failed])
                    curs.execute(sql,(save,author,dist,s_score,e_score,failed))
                conn.commit()
                conn.close()

            else:
                plotrunNolans2(final[['author_handle','s_score','e_score','Failed']])

    else:
        from plot_results import plotrunNolans
        vals = final.values[0]
        if print_results == True:
            print('Test User: {}\nModel Error: {}\nEconomic Score Estimate: {}\nSocial Score Estimate: {}'.format(
                vals[0], vals[1], vals[3], vals[2]))

        final.columns = ['author_handle', 'model_error',
                         's_score', 'e_score','Failed']
        if rec == True:
            return final
        else:
            plotrunNolans(final[['author_handle','s_score','e_score']],user)

def getRecommendations(user):
    final = main(user, print_results=False, rec=True)
    ## Use DFs for getting distances (model errors)
    final.columns = ['author_handle','error',0,1,'nonsense']
    final = final[[0,1]]
    compare_with = pd.DataFrame.from_dict(ground_truths_politicians, orient='index')
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
    best_loss = sorted_loss[:len(ground_truths_politicians.keys()) // 25]
    print('Hi {}! Based on your results, we recommend that you check out these Twitter users:\n{}\n'.format(user,
                                                                                 ', '.join([x[0] for x in best_loss])))

if __name__ == '__main__':
    for n in [5,10,15,25,30,40,50,60,70,80,90,100,120]:
        main(print_results=False, comps=n, save=n)
    #returnRecommendations('realDonaldTrump')
