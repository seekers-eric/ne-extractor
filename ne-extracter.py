# -*- coding: utf-8 -*-

import re
import ner
import json
import string
import nltk

from TfidfModel import TfidfModel
from TextrankModel import TextrankModel

stop_words = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

tagger = ner.SocketNER(host='localhost', port=8080)

text = '''Moments is a popular socialising function of WeChat, or Weixin as it is known in China, which allows users to network by sharing information, photos and articles with their friends privately.
The synchronisation function with Facebook and Twitter is already available to overseas WeChat users, and extending it to domestic users underlines Tencent’s efforts to expand its user base within and outside China.
Facebook and Twitter, whose active users have reached 1.97 billion and 319 million globally as of the end of April, remain inaccessible in the mainland. Tencent said over 937 million people are using its WeChat app to message friends.
The synchronisation function is now available to iPhone iOS users who had until now, only been able to synchronise their Moments contents to the another popular Tencent messaging service QQ. The function for Android users is expected to be available soon.
Facebook and Twitter have 1.97 billion and 319 million active users globally while Tencent says 937 million people are using WeChat to message friends. Photo: Reuters
Despite its huge investments to expand globally, Tencent was compelled to initially open up the synchronisation function to its overseas WeChat users as it faces difficulty breaking into the Western mainstream market where consumer habits were strongly entrenched, said Wang Xiaofeng, a senior analyst with Forrester Research.
“Now WeChat is more mature in its international strategy by focusing on Chinese tourists, Chinese overseas, and working with companies outside of China who want to reach Chinese consumers in China, as well as Chinese tourists,” Wang told the South China Morning Post.'''

translate_table = dict((ord(char), None) for char in string.punctuation)

tfidf_model = TfidfModel()
textrank_model = TextrankModel()


def replace_organization_phrase(raw):
    text = raw
    tags = json.loads(tagger.json_entities(raw))
    organizations = None
    if 'ORGANIZATION' in tags:
        organizations = tags['ORGANIZATION']
        replaceing_candidates = {}
        replacings = [o for o in organizations if ' ' in o]
        for replacing in replacings:
            replaced = re.sub(r'\s+', '-', replacing)
            replaceing_candidates[replaced] = replacing
            pattern = r'{0}'.format(replacing)
            organizations.remove(replacing)
            organizations.append(replaced)
            text = re.sub(pattern, replaced, text)

    return text, organizations

no_punctuation = text.translate(translate_table)
text, organizations = replace_organization_phrase(no_punctuation)
organizations = [x.lower() for x in organizations]
lowered = text.lower()
tokens = nltk.word_tokenize(lowered)
cleaned = [stemmer.stem(token) for token, pos in nltk.pos_tag(tokens)]

r_textrank = dict(textrank_model.score_keyphrases_by_textrank(' '.join(cleaned)))

oganization_list = list(set(organizations))

r_tfidf = {}
tfidf = tfidf_model.get_tfidf_model()
response = tfidf.transform([' '.join(oganization_list)])
print(response)
feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    organization = feature_names[col]
    score = response[0, col]
    print(organization + '-' + str(score))
    r_tfidf[organization] = score

r_final = []
for oganization in oganization_list:
    oganization = oganization.lower()
    textrank_score = r_textrank[oganization] if oganization in r_textrank else 0
    tfidf_score = 0
    if oganization in r_tfidf:
        times = organizations.count(oganization)
        tfidf_score = r_tfidf[organization] * times
        total_score = 2*(tfidf_score * textrank_score) / (tfidf_score + textrank_score)
    else:
        total_score = textrank_score
    r_final.append((oganization, total_score))

r_final = sorted(r_final, key=lambda x: x[1])
print(r_final)








