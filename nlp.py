import os
from konlpy.tag import Okt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
MODEL = os.path.join(ROOT, 'model')

okt = Okt()

