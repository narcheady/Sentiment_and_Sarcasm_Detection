import numpy as np
import pandas as pd
import json
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class ReviewClassifierService:

    words = {}

    sentiment_model = pickle.load(open('sentimentClassifier.pkl','rb'))
    sarcasm_model = pickle.load(open('sarcasmClassifier.pkl','rb'))

    def __init__(self):
        with open('word_feature_space.json', 'r') as fp:
            self.sentiment_words = json.load(fp)

        with open('sarcasm_word_feature_space.json', 'r') as sfp:
            self.sarcasm_words = json.load(sfp)

    #clean review get word roots, lowercase, split them by words and etc.
    def clean_review(self, review):
        review = review.lower()
        review = review.split()

        ps = PorterStemmer()

        review = [ps.stem(word) for word in review
                    if not word in set(stopwords.words('english'))]

        review = ' '.join(review)

        return review

    #return if the review was positive or negative as JSON.
    def sarcasm_classify(self, review):
        length = len(self.sarcasm_words)
        incoming = [0] * length
        clean_sample = self.clean_review(review)
        clean_words = re.sub("[^\w]", " ", clean_sample).split()

        for i in clean_words:
            if i in self.sarcasm_words:
                index = self.sarcasm_words[i]
                incoming[index] += 1

        outcome = self.sarcasm_model.predict([incoming])

        result = "sarcastic" if int(outcome[0]) == 1 else "not sarcastic"

        return json.dumps(result)

        
    #return if the review was positive or negative as JSON.
    def sentiment_classify(self, review):
        length = len(self.sentiment_words)
        incoming = [0] * length
        clean_sample = self.clean_review(review)
        clean_words = re.sub("[^\w]", " ", clean_sample).split()

        for i in clean_words:
            if i in self.sentiment_words:
                index = self.sentiment_words[i]
                incoming[index] += 1

        outcome = self.sentiment_model.predict([incoming])

        if int(outcome[0]) == -1:
            result = "positive"
        elif int(outcome[0]) == -1:
            result = "negative"
        else:
            result = "neutral"

        return json.dumps(result)