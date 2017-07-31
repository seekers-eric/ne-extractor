import os
import re
import ner
import json
import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

training_path = './training_data/tfidf_training_data.txt'
model_path = './model/tfidf.pkl'

tagger = ner.SocketNER(host='localhost', port=8080)

stop_words = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

translate_table = dict((ord(char), None) for char in string.punctuation)


def replace_organization_phrase(raw):
    text = ''
    tags = json.loads(tagger.json_entities(raw))
    if 'ORGANIZATION' in tags:
        organizations = tags['ORGANIZATION']
        replaceing_candidates = {}
        replacings = [o for o in organizations if ' ' in o]
        for replacing in replacings:
            replaced = re.sub(r'\s+', '-', replacing)
            replaceing_candidates[replaced] = replacing
            pattern = r'{0}'.format(replacing)
            text = re.sub(pattern, replaced, raw)
    else:
        text = raw

    return text


def tokenize(raw):
    lowered = replace_organization_phrase(raw).lower()
    no_punctuation = lowered.translate(translate_table)
    tokens = nltk.word_tokenize(no_punctuation)
    cleaned = [stemmer.stem(token) for token, pos in nltk.pos_tag(tokens)
               if token not in stop_words and pos.startswith('N')]
    return cleaned

if __name__ == '__main__':
    corpus = []
    tfidf = None
    if os.path.isfile(model_path):
        tfidf = joblib.load(model_path)
    else:
        with open(training_path, "r", encoding='utf-8') as fp:
            for line in fp:
                corpus.append(line)
            tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
            tfs = tfidf.fit_transform(corpus)

            joblib.dump(tfidf, model_path)

    text = 'Rather than Alphabet-Inc Facebook or Microsoft increasingly Chinese duo Alibaba and Tencent are the driving forces behind the importing of large sums of capital and vast business experience into Southeast Asia’s most promising startups'.lower()
    response = tfidf.transform([text])
    print(response)
    feature_names = tfidf.get_feature_names()
    for col in response.nonzero()[1]:
        print(feature_names[col], ' - ', response[0, col])
