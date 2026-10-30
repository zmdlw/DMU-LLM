import os
from konlpy.tag import Okt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
MODEL = os.path.join(ROOT, 'model')

okt = Okt()

WORD_TYPES = ['Noun', 'Verb', 'Adjective', 'Adverb', 'Exclamation']

def get_words(text):
    words_with_types = okt.pos(str(text), stem = True)
    words = []
    for word, word_type in words_with_types:
        if word_type in WORD_TYPES:
            words.append(word)
    return words