import re
import ner

from collections import Counter
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import CountVectorizer



tagger = ner.SocketNER(host='localhost', port=8080)

text = 'Rather than Alphabet Inc., Facebook or Microsoft, increasingly Chinese duo Alibaba and Tencent are the driving forces behind the importing of large sums of capital and vast business experience into Southeast Asia’s most promising startups.'

# tags = json.loads(tagger.json_entities(text))
# organizations = tags['ORGANIZATION']
#
# replaceing_candidates = {}
#
# replacings = [o for o in organizations if ' ' in o]
# for replacing in replacings:
#     replaced = re.sub(r'\s+', '-', replacing)
#     replaceing_candidates[replaced] = replacing
#     pattern = r'{0}'.format(replacing)
#     text = re.sub(pattern, replaced, text)

