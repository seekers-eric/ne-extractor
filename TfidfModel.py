# -*- coding: utf-8 -*-

import os
import re
import ner
import json
import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib


class TfidfModel:
    def __init__(self):
        self.training_path = './training_data/tfidf_training_data.txt'
        self.model_path = './model/tfidf.pkl'
        self.tagger = ner.SocketNER(host='localhost', port=8080)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.PorterStemmer()
        self.translate_table = dict((ord(char), None) for char in string.punctuation)

    def replace_organization_phrase(self, raw):
        text = raw
        tags = json.loads(self.tagger.json_entities(raw))
        if 'ORGANIZATION' in tags:
            organizations = tags['ORGANIZATION']
            replaceing_candidates = {}
            replacings = [o for o in organizations if ' ' in o]
            for replacing in replacings:
                replaced = re.sub(r'\s+', '-', replacing)
                replaceing_candidates[replaced] = replacing
                pattern = r'{0}'.format(replacing)
                text = re.sub(pattern, replaced, text)

        return text

    def tokenize_text(self, raw):
        lowered = self.replace_organization_phrase(raw).lower()
        no_punctuation = lowered.translate(self.translate_table)
        tokens = nltk.word_tokenize(no_punctuation)
        cleaned = [self.stemmer.stem(token) for token, pos in nltk.pos_tag(tokens)
                   if token not in self.stop_words and pos.startswith('N')]
        return cleaned

    def get_tfidf_model(self):
        tfidf = None
        if os.path.isfile(self.model_path):
            tfidf = joblib.load(self.model_path)
        else:
            corpus = []
            with open(self.training_path, "r", encoding='utf-8') as fp:
                for line in fp:
                    corpus.append(line)
                tfidf = TfidfVectorizer(tokenizer=self.tokenize_text, stop_words='english')
                tfidf.fit_transform(corpus)
                joblib.dump(tfidf, self.model_path)

        return tfidf

tfidf = TfidfModel()
model = tfidf.get_tfidf_model()
text = 'Rather than Alphabet Inc Facebook or Microsoft increasingly Chinese duo Alibaba and Tencent are the driving forces behind the importing of large sums of capital and vast business experience into Southeast Asia’s most promising startups'.lower()
response = model.transform([text])
print(response)
feature_names = model.get_feature_names()
for col in response.nonzero()[1]:
    print(feature_names[col], ' - ', response[0, col])
