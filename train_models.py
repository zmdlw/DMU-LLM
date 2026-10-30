import os, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nlp import DATA, MODEL, get_words

def train(csv_name, label, model_name, classifier, sublinear_tf=False):
    data = pd.read_csv(os.path.join(DATA, csv_name))
    train_x, test_x, train_y, test_y = train_test_split(data['문장'], data[label], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(tokenizer=get_words, token_pattern=None, ngram_range=(1, 2), sublinear_tf=sublinear_tf)
    classifier.fit(vectorizer.fit_transform(train_x), train_y)
    print(classification_report(test_y, classifier.predict(vectorizer.transform(test_x))))
    os.makedirs(MODEL, exist_ok=True)
    with open(os.path.join(MODEL, model_name), 'wb') as file:
        pickle.dump((vectorizer, classifier), file)

print("위기 감지 모델")
train('crisis_data.csv', '위기', 'crisis.pkl', LogisticRegression(class_weight='balanced', max_iter=1000))

print("감정 분류 모델")
train('emotion_data.csv', '감정', 'emotion.pkl', LogisticRegression(C=5, max_iter=1000), sublinear_tf=True)