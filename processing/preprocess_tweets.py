import sqlite3
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from re import sub


class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("english")

    def lemmatize_stemming(self, word):
        lemmatized = self.lemmatizer.lemmatize(word)
        return self.stemmer.stem(lemmatized)

    def preprocess(self, text):
        dirty_text = ["http\S+", "@\S+", "#", "&amp;"]
        ignore = [
            "thing",
            "called",
            "say",
            "good",
            "said",
            "go",
            "today",
            "day",
            "want",
            "need",
            "hear",
            "get",
            "like",
            "come",
            "week",
            "think",
            "thought",
            "month",
            "tonight",
            "year",
            "hear",
            "need",
            "know",
            "talk",
            "RT",
        ]

        # get rid of noisy words
        processed_ignores = []
        for word in ignore:
            processed_ignores.append(self.lemmatize_stemming(word))

        # get rid of links, referenced handles, hashtag signs (but not the actual hashtag text)
        for pattern in dirty_text:
            text = sub(r"{}".format(pattern), "", text)

        lemmed_and_stemmed = []
        for token in simple_preprocess(text):
            if token not in STOPWORDS and len(token) > 3:
                token = self.lemmatize_stemming(token)
                if token not in processed_ignores:
                    lemmed_and_stemmed.append(token)

        return lemmed_and_stemmed
