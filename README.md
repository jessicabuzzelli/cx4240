This is a project for CX 4240: Introduction to Computational Data Analysis at Georgia Tech built in Spring 2019. 

* presentation.ipynb is a jupyter notebook covering key results from our models and overall approach to identifying users' political alignment via comparing their tweets to politicians with known political affiliations.

* build_dataset pulls in tweets to a local SQLite database using a master dictionary mapping Twitter users to social and economic
"ground truth" scores from [OnTheIssues.org](). Twitter API keys and tokens can be obtained [here](https://developer.twitter.com/).
 
* preprocess_tweets can be applied to add a column to the database created by build_dataset
corresponding to tokenized, lemmatized, and stemmed words for each tweet in the database. 

* linear_regression_model's `main` function can accept a user's Twitter handle as a parameter to find a Nolan Chart point
estimation based on TF-IDF vectorized data (xtrain from dataset politicians' tweets, xtest from user's tweets). User must 
be present in the database at this time. Primary Component Analysis is applied prior to the regression to stabilize results
via the number of features to less than that of training users (currently 75 and 129, respectively).

* LDA_model runs a Latent Dirichlet Allocation to reduce the number of features of TF-IDF vectorized data prior to a linear regression mapping a user to its Nolan Chart position. There is an option to user LSA instead.

* LDA_sentiment is a messy version of LDA_model that obtains a feature vectors for linear regression via determining the main topic of a tweet, mapping it to a its sentiment as determined by running [Vader](https://www.nltk.org/_modules/nltk/sentiment/vader.html),
and obtaining a user's average sentiment towards each LDA topic. LSI_sentiment is similarly messy and slow, but applied truncated SVD to determine topics. Sentiment models are extrememly volatile and ill-advised.

* plot_results is a program built to support the models rather than run independently.

This project respected Twitter's advisory on using its data to classify users based on a federally protected class such 
as political affiliation via only using tweets from elected officials or professional political commentators in our 
dataset and models. 

Our estimates tend to assign users to the "centrist" area on the Nolan Chart and our methods could 
be improved via implementing ensemble modeling, abandoning regression altogether via a random forest classifier, or
reframing the project to a clustering over a continuum defined by the most extreme training users. 
