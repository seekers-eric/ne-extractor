# -*- coding: utf-8 -*-

import re
import ner
import json
import string
import nltk

from TfidfModel import TfidfModel

stop_words = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

tagger = ner.SocketNER(host='localhost', port=8080)

text = 'Rather than Alphabet Inc., Facebook or Microsoft, increasingly Chinese duo Alibaba and Tencent are the driving forces behind the importing of large sums of capital and vast business experience into Southeast Asia’s most promising startups.'

translate_table = dict((ord(char), None) for char in string.punctuation)

tfidf_model = TfidfModel()


def replace_organization_phrase(raw):
    text = raw
    tags = json.loads(tagger.json_entities(raw))
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

lowered = replace_organization_phrase(text).lower()
no_punctuation = lowered.translate(translate_table)
tokens = nltk.word_tokenize(no_punctuation)
cleaned = [stemmer.stem(token) for token, pos in nltk.pos_tag(tokens)
           if token not in stop_words and pos.startswith('N')]

tfidf = tfidf_model.get_tfidf_model()
response = tfidf.transform([text])
print(response)
feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print(feature_names[col], ' - ', response[0, col])

