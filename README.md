This is a project for CX 4240: Computational Data Analysis at Georgia Tech, taught in Spring 2019 by Dr. Mahdi Roozbahani. 

`final_presentation` steps through my approach to identifying users' political alignment via comparing their tweets to politicians with known political affiliations and my final analysis.

This project used a custom dataset of tweets from well-known politicians and aimed to determine language patterns that could be used to justify their "ground truth" Nolan chart alignment scores obtained from [OnTheIssues.org]() in March 2018. Twitter API keys and tokens were obtained [here](https://developer.twitter.com/).

* `linear_regression_model` outputs a Nolan alignment estimation based on TF-IDF vectorized data. Primary Component Analysis is applied prior to the regression to stabilize the number of features to less than that of training users.

* `topic_model` runs a Latent Dirichlet Allocation to reduce the number of features of TF-IDF vectorized data prior to a linear regression mapping a user to its Nolan Chart position. There is an option to use LSA instead, but differences in training accuracy were insignificant.

* `topic_sentiment_model` builds on `topic_model` by determining the main topic of a tweet, mapping it to a its sentiment as determined by running [Vader](https://www.nltk.org/_modules/nltk/sentiment/vader.html),
and obtaining a user's average sentiment towards each LDA topic. Sentiment-conscious models produced extremely volatile results in testing.

This project respected Twitter's advisory on using its data to classify users based on a federally protected class such 
as political affiliation via only using tweets from elected officials or professional political commentators in our 
dataset and models. 

My estimates tend to assign users to the "centrist" area on the Nolan Chart and the methods could 
be improved via implementing ensemble modeling, abandoning regression altogether via a random forest classifier, or
reframing the project to an unsupervised clustering over a continuum defined by the most extreme training users. 
