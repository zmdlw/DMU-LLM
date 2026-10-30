import os, pickle
from nlp import MODEL, get_words

MODEL = os.path.join(MODEL, 'emotion.pkl')

model = None

def predict(text, history=()):
    global model
    if model is None:
        model = pickle.load(open(MODEL, 'rb'))
    vectorizer, classifier = model
    probabilities = classifier.predict_proba(vectorizer.transform([text]))[0]
    if history and probabilities.max() < 0.4:
        last_probabilities = classifier.predict_proba(vectorizer.transform([history[-1]]))[0]
        probabilities = probabilities * 0.7 + last_probabilities * 0.3
    best, second = probabilities.argsort()[::-1][:2] 
    keywords = []
    for word in get_words(text):
        if word in vectorizer.vocabulary_ and word not in keywords:
            keywords.append(word)
    return {'emotion': classifier.classes_[best], 'probability': probabilities[best], 'keywords': keywords, 'second_emotion': classifier.classes_[second]}